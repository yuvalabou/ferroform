//! Pure-Rust manifold CSG utilities: shell void subtraction and Z-plane flat-capping.

use std::collections::HashMap;

use crate::manifold::ensure_positive_volume;
use crate::marching::MeshData;

/// Subtract an interior void mesh from an outer watertight shell mesh.
///
/// Both `outer` and `void_mesh` are oriented with positive outward signed volume;
/// `void_mesh` is appended with inverted face winding so the resulting cavity
/// faces inward and preserves all original outer CAD triangles intact.
pub fn subtract_interior_void(outer: &MeshData, void_mesh: &MeshData) -> MeshData {
    let mut result = outer.clone();
    ensure_positive_volume(&mut result);

    if void_mesh.points.is_empty() || void_mesh.faces.is_empty() {
        return result;
    }

    let mut void_oriented = void_mesh.clone();
    ensure_positive_volume(&mut void_oriented);

    let v_offset = result.points.len() as u32;
    result.points.extend_from_slice(&void_oriented.points);
    result.faces.reserve(void_oriented.faces.len());
    for f in &void_oriented.faces {
        // Flip winding so normals point inward into the subtracted cavity.
        result
            .faces
            .push([f[0] + v_offset, f[2] + v_offset, f[1] + v_offset]);
    }
    result
}

/// Trim a closed mesh by top and bottom Z planes (1% margin) and cap the cut faces.
pub fn trim_flat_cap(mesh: &MeshData) -> MeshData {
    if mesh.points.is_empty() || mesh.faces.is_empty() {
        return mesh.clone();
    }

    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for p in &mesh.points {
        let z = p[2] as f64;
        z_min = z_min.min(z);
        z_max = z_max.max(z);
    }

    let margin = (z_max - z_min) * 0.01;
    if margin <= 0.001 {
        return mesh.clone();
    }

    let top_cut = trim_by_z_plane(mesh, z_max - margin, true);
    if top_cut.points.is_empty() || top_cut.faces.is_empty() {
        return top_cut;
    }
    trim_by_z_plane(&top_cut, z_min + margin, false)
}

/// Slice a closed, consistently oriented mesh against `z = z_cut` and cap the cut loops.
///
/// When `keep_below` is `true`, keeps `z <= z_cut` and caps with `+Z` normal.
/// When `keep_below` is `false`, keeps `z >= z_cut` and caps with `-Z` normal.
pub fn trim_by_z_plane(mesh: &MeshData, z_cut: f64, keep_below: bool) -> MeshData {
    if mesh.points.is_empty() || mesh.faces.is_empty() {
        return MeshData::default();
    }

    // Nudge the cut plane slightly if any vertex sits on it to avoid coplanar degeneracies.
    let mut zc = z_cut;
    for _ in 0..16 {
        if mesh
            .points
            .iter()
            .all(|p| ((p[2] as f64) - zc).abs() >= 1e-6)
        {
            break;
        }
        zc += if keep_below { -1.37e-6 } else { 1.37e-6 };
    }

    let kept: Vec<bool> = mesh
        .points
        .iter()
        .map(|p| {
            let z = p[2] as f64;
            if keep_below {
                z <= zc
            } else {
                z >= zc
            }
        })
        .collect();

    if kept.iter().all(|&k| k) {
        return mesh.clone();
    }
    if kept.iter().all(|&k| !k) {
        return MeshData::default();
    }

    let mut new_points: Vec<[f32; 3]> = Vec::with_capacity(mesh.points.len());
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut vert_map = vec![u32::MAX; mesh.points.len()];
    let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut cap_segments: Vec<(u32, u32)> = Vec::new();

    let mut get_orig = |idx: u32, pts: &mut Vec<[f32; 3]>| -> u32 {
        let mapped = vert_map[idx as usize];
        if mapped != u32::MAX {
            return mapped;
        }
        let new_idx = pts.len() as u32;
        pts.push(mesh.points[idx as usize]);
        vert_map[idx as usize] = new_idx;
        new_idx
    };

    let mut get_inter = |a: u32, b: u32, pts: &mut Vec<[f32; 3]>| -> u32 {
        let key = (a.min(b), a.max(b));
        if let Some(&idx) = edge_map.get(&key) {
            return idx;
        }
        let pa = mesh.points[a as usize];
        let pb = mesh.points[b as usize];
        let za = pa[2] as f64;
        let zb = pb[2] as f64;
        let denom = zb - za;
        let t = if denom.abs() < 1e-18 {
            0.5
        } else {
            ((zc - za) / denom).clamp(0.0, 1.0)
        };
        let ix = (pa[0] as f64) + t * ((pb[0] as f64) - (pa[0] as f64));
        let iy = (pa[1] as f64) + t * ((pb[1] as f64) - (pa[1] as f64));
        let new_idx = pts.len() as u32;
        pts.push([ix as f32, iy as f32, zc as f32]);
        edge_map.insert(key, new_idx);
        new_idx
    };

    for &f in &mesh.faces {
        let k = [
            kept[f[0] as usize],
            kept[f[1] as usize],
            kept[f[2] as usize],
        ];
        let count = k.iter().filter(|&&b| b).count();
        match count {
            3 => {
                let a = get_orig(f[0], &mut new_points);
                let b = get_orig(f[1], &mut new_points);
                let c = get_orig(f[2], &mut new_points);
                new_faces.push([a, b, c]);
            }
            0 => {}
            1 => {
                // Rotate so `a` is kept and `b, c` are clipped.
                let rot = if k[0] {
                    0
                } else if k[1] {
                    1
                } else {
                    2
                };
                let (a, b, c) = (f[rot], f[(rot + 1) % 3], f[(rot + 2) % 3]);
                let na = get_orig(a, &mut new_points);
                let p_ab = get_inter(a, b, &mut new_points);
                let p_ac = get_inter(a, c, &mut new_points);
                new_faces.push([na, p_ab, p_ac]);
                // Kept triangle traverses p_ab -> p_ac; cap must traverse p_ac -> p_ab.
                cap_segments.push((p_ac, p_ab));
            }
            2 => {
                // Rotate so `a, b` are kept and `c` is clipped.
                let rot = if !k[2] {
                    0
                } else if !k[0] {
                    1
                } else {
                    2
                };
                let (a, b, c) = (f[rot], f[(rot + 1) % 3], f[(rot + 2) % 3]);
                let na = get_orig(a, &mut new_points);
                let nb = get_orig(b, &mut new_points);
                let p_bc = get_inter(b, c, &mut new_points);
                let p_ca = get_inter(c, a, &mut new_points);
                new_faces.push([na, nb, p_bc]);
                new_faces.push([na, p_bc, p_ca]);
                // Kept triangle traverses p_bc -> p_ca; cap must traverse p_ca -> p_bc.
                cap_segments.push((p_ca, p_bc));
            }
            _ => unreachable!(),
        }
    }

    if !cap_segments.is_empty() {
        let loops = chain_segments(&cap_segments, &new_points, keep_below);
        triangulate_loops(&loops, &new_points, keep_below, &mut new_faces);
    }

    MeshData {
        points: new_points,
        faces: new_faces,
    }
}

#[inline]
fn to_2d(p: [f32; 3], keep_below: bool) -> [f64; 2] {
    if keep_below {
        [p[0] as f64, p[1] as f64]
    } else {
        [p[0] as f64, -(p[1] as f64)]
    }
}

#[inline]
fn cross2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn signed_area_2d(loop_verts: &[u32], points: &[[f32; 3]], keep_below: bool) -> f64 {
    let n = loop_verts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area2 = 0.0;
    for i in 0..n {
        let a = to_2d(points[loop_verts[i] as usize], keep_below);
        let b = to_2d(points[loop_verts[(i + 1) % n] as usize], keep_below);
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    area2 * 0.5
}

fn chain_segments(
    segments: &[(u32, u32)],
    points: &[[f32; 3]],
    keep_below: bool,
) -> Vec<Vec<u32>> {
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::with_capacity(segments.len());
    for &(u, v) in segments {
        if u != v {
            adj.entry(u).or_default().push(v);
        }
    }

    let mut keys: Vec<u32> = adj.keys().copied().collect();
    keys.sort_unstable();

    let mut loops = Vec::new();
    for start in keys {
        loop {
            let next_opt = adj.get_mut(&start).and_then(|v| v.pop());
            let Some(mut curr) = next_opt else {
                break;
            };
            let mut prev = start;
            let mut poly = vec![start];

            while curr != start {
                poly.push(curr);
                let Some(cands) = adj.get_mut(&curr) else {
                    break;
                };
                if cands.is_empty() {
                    break;
                }
                let next = if cands.len() == 1 {
                    cands.pop().unwrap()
                } else {
                    // At a pinch vertex, pick the sharpest right-hand turn in 2D.
                    let p_prev = to_2d(points[prev as usize], keep_below);
                    let p_curr = to_2d(points[curr as usize], keep_below);
                    let in_x = p_prev[0] - p_curr[0];
                    let in_y = p_prev[1] - p_curr[1];
                    let mut best_i = 0;
                    let mut best_ang = f64::INFINITY;
                    for (ci, &cand) in cands.iter().enumerate() {
                        let p_cand = to_2d(points[cand as usize], keep_below);
                        let ox = p_cand[0] - p_curr[0];
                        let oy = p_cand[1] - p_curr[1];
                        let cross = in_x * oy - in_y * ox;
                        let dot = in_x * ox + in_y * oy;
                        let mut ang = cross.atan2(dot);
                        if ang <= 0.0 {
                            ang += std::f64::consts::TAU;
                        }
                        if ang < best_ang {
                            best_ang = ang;
                            best_i = ci;
                        }
                    }
                    cands.swap_remove(best_i)
                };
                prev = curr;
                curr = next;
                if poly.len() > segments.len() + 2 {
                    break;
                }
            }

            if curr == start && poly.len() >= 3 {
                loops.push(poly);
            }
        }
    }
    loops
}

fn point_in_poly_2d(pt: [f64; 2], poly: &[u32], points: &[[f32; 3]], keep_below: bool) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = to_2d(points[poly[i] as usize], keep_below);
        let b = to_2d(points[poly[j] as usize], keep_below);
        if ((a[1] > pt[1]) != (b[1] > pt[1]))
            && (pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1]) + a[0])
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn triangulate_loops(
    loops: &[Vec<u32>],
    points: &[[f32; 3]],
    keep_below: bool,
    out_faces: &mut Vec<[u32; 3]>,
) {
    let mut outers: Vec<(usize, f64)> = Vec::new();
    let mut holes: Vec<(usize, f64)> = Vec::new();

    for (li, lp) in loops.iter().enumerate() {
        let area = signed_area_2d(lp, points, keep_below);
        if area.abs() < 1e-14 {
            continue;
        }
        if area > 0.0 {
            outers.push((li, area));
        } else {
            holes.push((li, area));
        }
    }

    let mut assigned_holes: Vec<Vec<usize>> = vec![Vec::new(); outers.len()];
    for &(hi, _) in &holes {
        let h_pt = to_2d(points[loops[hi][0] as usize], keep_below);
        let mut best_outer: Option<(usize, f64)> = None;
        for (oi, &(outer_li, outer_area)) in outers.iter().enumerate() {
            if point_in_poly_2d(h_pt, &loops[outer_li], points, keep_below)
                && best_outer.map_or(true, |(_, ba)| outer_area < ba)
            {
                best_outer = Some((oi, outer_area));
            }
        }
        if let Some((oi, _)) = best_outer {
            assigned_holes[oi].push(hi);
        }
    }

    for (oi, &(outer_li, _)) in outers.iter().enumerate() {
        let mut poly = loops[outer_li].clone();
        if !assigned_holes[oi].is_empty() {
            let mut hole_list: Vec<(usize, f64)> = assigned_holes[oi]
                .iter()
                .map(|&hi| {
                    let max_x = loops[hi]
                        .iter()
                        .map(|&vi| to_2d(points[vi as usize], keep_below)[0])
                        .fold(f64::NEG_INFINITY, f64::max);
                    (hi, max_x)
                })
                .collect();
            hole_list.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (hi, _) in hole_list {
                poly = bridge_hole(poly, &loops[hi], points, keep_below);
            }
        }
        ear_clip_simple(&poly, points, keep_below, out_faces);
    }
}

fn bridge_hole(
    mut outer: Vec<u32>,
    hole: &[u32],
    points: &[[f32; 3]],
    keep_below: bool,
) -> Vec<u32> {
    let mut h_pos = 0usize;
    let mut h_max_x = f64::NEG_INFINITY;
    for (i, &vi) in hole.iter().enumerate() {
        let x = to_2d(points[vi as usize], keep_below)[0];
        if x > h_max_x {
            h_max_x = x;
            h_pos = i;
        }
    }
    let h_pt = to_2d(points[hole[h_pos] as usize], keep_below);

    // Ray-cast +X from h_pt to find closest visible edge on outer polygon.
    let n = outer.len();
    let mut best_t = f64::INFINITY;
    let mut best_edge = 0usize;
    for i in 0..n {
        let a = to_2d(points[outer[i] as usize], keep_below);
        let b = to_2d(points[outer[(i + 1) % n] as usize], keep_below);
        if (a[1] <= h_pt[1] && b[1] > h_pt[1]) || (b[1] <= h_pt[1] && a[1] > h_pt[1]) {
            let t = (h_pt[1] - a[1]) / (b[1] - a[1]);
            let ix = a[0] + t * (b[0] - a[0]);
            let dist = ix - h_pt[0];
            if dist >= -1e-12 && dist < best_t {
                best_t = dist;
                best_edge = i;
            }
        }
    }

    if !best_t.is_finite() {
        // Fallback: connect to nearest outer vertex.
        let mut nearest = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (i, &vi) in outer.iter().enumerate() {
            let p = to_2d(points[vi as usize], keep_below);
            let d2 = (p[0] - h_pt[0]).powi(2) + (p[1] - h_pt[1]).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                nearest = i;
            }
        }
        return splice_bridge(&mut outer, nearest, hole, h_pos);
    }

    let i_pt = [h_pt[0] + best_t, h_pt[1]];
    let a_idx = best_edge;
    let b_idx = (best_edge + 1) % n;
    let pa = to_2d(points[outer[a_idx] as usize], keep_below);
    let pb = to_2d(points[outer[b_idx] as usize], keep_below);
    let mut v_pos = if pa[0] >= pb[0] { a_idx } else { b_idx };
    let mut p_pt = to_2d(points[outer[v_pos] as usize], keep_below);

    // Check reflex outer vertices inside triangle (h_pt, i_pt, p_pt).
    let tri_sign = cross2d(h_pt, i_pt, p_pt);
    let mut best_cos = ((p_pt[0] - h_pt[0])
        / ((p_pt[0] - h_pt[0]).hypot(p_pt[1] - h_pt[1]).max(1e-15)))
    .clamp(-1.0, 1.0);

    if tri_sign.abs() > 1e-14 {
        for r in 0..n {
            if r == v_pos {
                continue;
            }
            let rp = to_2d(points[outer[r] as usize], keep_below);
            if rp[0] < h_pt[0] - 1e-12 {
                continue;
            }
            let r_prev = to_2d(points[outer[(r + n - 1) % n] as usize], keep_below);
            let r_next = to_2d(points[outer[(r + 1) % n] as usize], keep_below);
            if cross2d(r_prev, rp, r_next) > 1e-12 {
                continue; // Not reflex.
            }
            if point_in_tri_oriented(rp, h_pt, i_pt, p_pt, tri_sign > 0.0) {
                let len = (rp[0] - h_pt[0]).hypot(rp[1] - h_pt[1]).max(1e-15);
                let cos_ang = (rp[0] - h_pt[0]) / len;
                if cos_ang > best_cos {
                    best_cos = cos_ang;
                    v_pos = r;
                    p_pt = rp;
                }
            }
        }
    }
    let _ = p_pt;

    splice_bridge(&mut outer, v_pos, hole, h_pos)
}

fn splice_bridge(outer: &mut Vec<u32>, v_pos: usize, hole: &[u32], h_pos: usize) -> Vec<u32> {
    let mut merged = Vec::with_capacity(outer.len() + hole.len() + 2);
    merged.extend_from_slice(&outer[..=v_pos]);
    merged.extend_from_slice(&hole[h_pos..]);
    merged.extend_from_slice(&hole[..=h_pos]);
    merged.push(outer[v_pos]);
    merged.extend_from_slice(&outer[v_pos + 1..]);
    merged
}

#[inline]
fn point_in_tri_oriented(
    p: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    ccw: bool,
) -> bool {
    let s1 = cross2d(a, b, p);
    let s2 = cross2d(b, c, p);
    let s3 = cross2d(c, a, p);
    if ccw {
        s1 >= -1e-12 && s2 >= -1e-12 && s3 >= -1e-12
    } else {
        s1 <= 1e-12 && s2 <= 1e-12 && s3 <= 1e-12
    }
}

fn ear_clip_simple(
    poly_in: &[u32],
    points: &[[f32; 3]],
    keep_below: bool,
    out_faces: &mut Vec<[u32; 3]>,
) {
    if poly_in.len() < 3 {
        return;
    }
    let mut poly = poly_in.to_vec();

    while poly.len() > 3 {
        let n = poly.len();
        let mut clipped = false;

        'pass: for tol in [1e-12_f64, -1e-10_f64] {
            for i in 0..n {
                let prev = poly[(i + n - 1) % n];
                let curr = poly[i];
                let next = poly[(i + 1) % n];
                let a = to_2d(points[prev as usize], keep_below);
                let b = to_2d(points[curr as usize], keep_below);
                let c = to_2d(points[next as usize], keep_below);

                if cross2d(a, b, c) <= tol {
                    continue;
                }

                let mut contains_other = false;
                for &q in &poly {
                    if q == prev || q == curr || q == next {
                        continue;
                    }
                    let qp = to_2d(points[q as usize], keep_below);
                    if point_in_tri_oriented(qp, a, b, c, true) {
                        contains_other = true;
                        break;
                    }
                }
                if !contains_other {
                    out_faces.push([prev, curr, next]);
                    poly.remove(i);
                    clipped = true;
                    break 'pass;
                }
            }
        }

        if !clipped {
            // Guaranteed-termination fallback for degenerate/self-touching loops.
            out_faces.push([poly[0], poly[1], poly[2]]);
            poly.remove(1);
        }
    }

    if poly.len() == 3 {
        out_faces.push([poly[0], poly[1], poly[2]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::signed_volume;

    fn make_box(x0: f32, x1: f32, y0: f32, y1: f32, z0: f32, z1: f32) -> MeshData {
        let points = vec![
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        MeshData { points, faces }
    }

    fn count_boundary_edges(mesh: &MeshData) -> usize {
        let mut counts: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &mesh.faces {
            for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *counts.entry((e.0, e.1)).or_insert(0) += 1;
                *counts.entry((e.1, e.0)).or_insert(0) -= 1;
            }
        }
        counts.values().filter(|&&c| c != 0).count()
    }

    #[test]
    fn test_subtract_interior_void_and_flat_cap() {
        let outer = make_box(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0);
        let inner = make_box(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
        let shell = subtract_interior_void(&outer, &inner);

        assert_eq!(count_boundary_edges(&shell), 0);
        let vol = signed_volume(&shell);
        assert!((vol - (1000.0 - 64.0)).abs() < 1e-2, "vol={vol}");

        // Trim flat_cap: 1% margin on height 10 -> 0.1 from top and bottom -> new height 9.8
        let capped = trim_flat_cap(&shell);
        assert_eq!(count_boundary_edges(&capped), 0);
        let capped_vol = signed_volume(&capped);
        assert!((capped_vol - (980.0 - 64.0)).abs() < 1e-1, "capped_vol={capped_vol}");
    }

    #[test]
    fn test_trim_by_z_plane_slices_through_hole() {
        let outer = make_box(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0);
        let inner = make_box(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
        let shell = subtract_interior_void(&outer, &inner);

        // Slice right through the middle of the internal void at z = 0.0
        let half = trim_by_z_plane(&shell, 0.0, true);
        assert_eq!(count_boundary_edges(&half), 0);
        let vol = signed_volume(&half);
        // Bottom half: 10*10*5 - 4*4*2 = 500 - 32 = 468
        assert!((vol - 468.0).abs() < 1e-1, "vol={vol}");
    }
}
