//! Welded parallel Marching Cubes isosurface extraction for Cartesian and cylindrical grids.

use rayon::prelude::*;

use crate::mc_tables::{EDGE_TABLE, TRI_TABLE};

/// Extracted indexed triangle mesh.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub points: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

impl MeshData {
    #[inline]
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn n_cells(&self) -> usize {
        self.faces.len()
    }

    /// Flat XYZ coordinate buffer `[x0, y0, z0, x1, y1, z1, ...]` for PyVista/NumPy handoff.
    pub fn points_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.points.len() * 3);
        for p in &self.points {
            flat.push(p[0]);
            flat.push(p[1]);
            flat.push(p[2]);
        }
        flat
    }

    /// VTK PolyData cell connectivity buffer `[3, v0, v1, v2, ...]`.
    pub fn vtk_cells(&self) -> Vec<i64> {
        let mut cells = Vec::with_capacity(self.faces.len() * 4);
        for f in &self.faces {
            cells.push(3);
            cells.push(f[0] as i64);
            cells.push(f[1] as i64);
            cells.push(f[2] as i64);
        }
        cells
    }

    /// Remove vertices not referenced by any triangle and remap face indices.
    pub fn compact_unreferenced(&mut self) {
        if self.faces.is_empty() {
            self.points.clear();
            return;
        }
        let mut used = vec![u32::MAX; self.points.len()];
        let mut new_points = Vec::with_capacity(self.points.len());
        for f in &mut self.faces {
            for idx in f.iter_mut() {
                let old = *idx as usize;
                let mapped = used[old];
                if mapped == u32::MAX {
                    let next_id = new_points.len() as u32;
                    used[old] = next_id;
                    new_points.push(self.points[old]);
                    *idx = next_id;
                } else {
                    *idx = mapped;
                }
            }
        }
        self.points = new_points;
    }
}

/// Grid coordinate mapping supporting both Cartesian and curvi-cylindrical grids.
pub enum GridCoords<'a> {
    Cartesian {
        xs: &'a [f64],
        ys: &'a [f64],
        zs: &'a [f64],
    },
    Cylindrical {
        rs: &'a [f64],
        phi_cs: Vec<(f64, f64)>,
        zs: &'a [f64],
    },
}

impl<'a> GridCoords<'a> {
    #[inline]
    fn dims(&self) -> (usize, usize, usize) {
        match self {
            Self::Cartesian { xs, ys, zs } => (xs.len(), ys.len(), zs.len()),
            Self::Cylindrical { rs, phi_cs, zs } => (rs.len(), phi_cs.len(), zs.len()),
        }
    }

    #[inline]
    fn pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        match self {
            Self::Cartesian { xs, ys, zs } => [xs[i], ys[j], zs[k]],
            Self::Cylindrical { rs, phi_cs, zs } => {
                let r = rs[i];
                let (c, s) = phi_cs[j];
                [r * c, r * s, zs[k]]
            }
        }
    }
}

#[inline]
fn safe_val(v: f64) -> f64 {
    if v >= 0.0 && v < 1e-6 {
        1e-6
    } else if v < 0.0 && v > -1e-6 {
        -1e-6
    } else {
        v
    }
}

/// Extract zero-level isosurface from a Cartesian grid `(xs, ys, zs)` in C-order.
pub fn extract_isosurface_cartesian(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    field: &[f64],
) -> MeshData {
    extract_isosurface(&GridCoords::Cartesian { xs, ys, zs }, field)
}

/// Extract zero-level isosurface from a curvi-cylindrical grid `(rs, phis, zs, ky)` in C-order.
pub fn extract_isosurface_cylindrical(
    rs: &[f64],
    phis: &[f64],
    zs: &[f64],
    ky: f64,
    field: &[f64],
) -> MeshData {
    let phi_cs: Vec<(f64, f64)> = phis
        .iter()
        .map(|&phi| {
            let angle = phi * ky;
            (angle.cos(), angle.sin())
        })
        .collect();
    extract_isosurface(&GridCoords::Cylindrical { rs, phi_cs, zs }, field)
}

const HI_BIT: u32 = 1 << 31;

/// Compact open-addressed hash map for sparse `(i, j, k, axis)` -> `vert_id` edge welding.
struct SparseEdgeMap {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
    len: usize,
}

impl SparseEdgeMap {
    #[inline]
    fn hash(mut x: u64) -> usize {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (x ^ (x >> 31)) as usize
    }

    fn with_capacity(expected_items: usize) -> Self {
        let cap = (expected_items.saturating_mul(2)).next_power_of_two().max(1024);
        Self {
            keys: vec![u64::MAX; cap],
            vals: vec![0; cap],
            mask: cap - 1,
            len: 0,
        }
    }

    #[inline]
    fn get(&self, key: u64) -> Option<u32> {
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let k = self.keys[idx];
            if k == key {
                return Some(self.vals[idx]);
            }
            if k == u64::MAX {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let new_cap = self.keys.len() * 2;
        let new_mask = new_cap - 1;
        let mut new_keys = vec![u64::MAX; new_cap];
        let mut new_vals = vec![0; new_cap];
        for (&k, &v) in self.keys.iter().zip(self.vals.iter()) {
            if k != u64::MAX {
                let mut idx = Self::hash(k) & new_mask;
                while new_keys[idx] != u64::MAX {
                    idx = (idx + 1) & new_mask;
                }
                new_keys[idx] = k;
                new_vals[idx] = v;
            }
        }
        self.keys = new_keys;
        self.vals = new_vals;
        self.mask = new_mask;
    }

    #[inline]
    fn insert(&mut self, key: u64, val: u32) {
        if (self.len + 1) * 2 >= self.keys.len() {
            self.grow();
        }
        let mut idx = Self::hash(key) & self.mask;
        loop {
            let k = self.keys[idx];
            if k == u64::MAX {
                self.keys[idx] = key;
                self.vals[idx] = val;
                self.len += 1;
                return;
            }
            if k == key {
                self.vals[idx] = val;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

struct ChunkResult {
    edge_map: SparseEdgeMap,
    owned_points: Vec<[f32; 3]>,
    hi_points: Vec<[f32; 3]>,
    hi_slots: Vec<u64>,
    faces: Vec<[u32; 3]>,
    skipped_degenerate: bool,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn get_or_insert_edge(
    coords: &GridCoords<'_>,
    field: &[f64],
    ny: usize,
    nz: usize,
    stride_x: usize,
    i_end: usize,
    is_last_chunk: bool,
    edge_map: &mut SparseEdgeMap,
    owned_points: &mut Vec<[f32; 3]>,
    hi_points: &mut Vec<[f32; 3]>,
    hi_slots: &mut Vec<u64>,
    gi: usize,
    gj: usize,
    gk: usize,
    axis: usize,
) -> u32 {
    let slot_idx = (((gi * ny + gj) * nz + gk) * 3 + axis) as u64;
    if let Some(existing) = edge_map.get(slot_idx) {
        return existing;
    }

    let (ni, nj, nk) = match axis {
        0 => (gi + 1, gj, gk),
        1 => (gi, gj + 1, gk),
        _ => (gi, gj, gk + 1),
    };
    let v0 = safe_val(field[gi * stride_x + gj * nz + gk]);
    let v1 = safe_val(field[ni * stride_x + nj * nz + nk]);
    let denom = v1 - v0;
    let mu = if denom.abs() < 1e-18 {
        0.5
    } else {
        (-v0 / denom).clamp(1e-4, 1.0 - 1e-4)
    };

    let p0 = coords.pos(gi, gj, gk);
    let p1 = coords.pos(ni, nj, nk);
    let pt = [
        (p0[0] + mu * (p1[0] - p0[0])) as f32,
        (p0[1] + mu * (p1[1] - p0[1])) as f32,
        (p0[2] + mu * (p1[2] - p0[2])) as f32,
    ];

    let vert_id = if !is_last_chunk && gi == i_end {
        let hi_id = hi_points.len() as u32;
        hi_points.push(pt);
        hi_slots.push(slot_idx);
        HI_BIT | hi_id
    } else {
        let id = owned_points.len() as u32;
        owned_points.push(pt);
        id
    };
    edge_map.insert(slot_idx, vert_id);
    vert_id
}

#[inline(always)]
fn chunk_point(v: u32, owned_points: &[[f32; 3]], hi_points: &[[f32; 3]]) -> [f32; 3] {
    if (v & HI_BIT) != 0 {
        hi_points[(v & !HI_BIT) as usize]
    } else {
        owned_points[v as usize]
    }
}

fn split_by_lengths<'a, T>(
    mut slice: &'a mut [T],
    lengths: impl Iterator<Item = usize>,
) -> Vec<&'a mut [T]> {
    let mut out = Vec::new();
    for len in lengths {
        let (head, tail) = slice.split_at_mut(len);
        out.push(head);
        slice = tail;
    }
    out
}

/// Extract a welded, 2-manifold zero-level isosurface from a 3D scalar field.
pub fn extract_isosurface(coords: &GridCoords<'_>, field: &[f64]) -> MeshData {
    let (nx, ny, nz) = coords.dims();
    if nx < 2 || ny < 2 || nz < 2 || field.len() != nx * ny * nz {
        return MeshData::default();
    }

    let stride_x = ny * nz;
    let stride_y = nz;

    // Step 1: Identify active cells in parallel across i-slabs.
    let active_slabs: Vec<Vec<(usize, usize, usize, u8)>> = (0..nx - 1)
        .into_par_iter()
        .map(|i| {
            let mut local = Vec::new();
            for j in 0..ny - 1 {
                let base00 = i * stride_x + j * stride_y;
                let base10 = base00 + stride_x;
                let base01 = base00 + stride_y;
                let base11 = base10 + stride_y;

                for k in 0..nz - 1 {
                    let mut cube_index = 0u8;
                    if safe_val(field[base00 + k]) < 0.0 {
                        cube_index |= 1;
                    }
                    if safe_val(field[base10 + k]) < 0.0 {
                        cube_index |= 2;
                    }
                    if safe_val(field[base11 + k]) < 0.0 {
                        cube_index |= 4;
                    }
                    if safe_val(field[base01 + k]) < 0.0 {
                        cube_index |= 8;
                    }
                    if safe_val(field[base00 + k + 1]) < 0.0 {
                        cube_index |= 16;
                    }
                    if safe_val(field[base10 + k + 1]) < 0.0 {
                        cube_index |= 32;
                    }
                    if safe_val(field[base11 + k + 1]) < 0.0 {
                        cube_index |= 64;
                    }
                    if safe_val(field[base01 + k + 1]) < 0.0 {
                        cube_index |= 128;
                    }

                    if cube_index != 0 && cube_index != 255 {
                        local.push((i, j, k, cube_index));
                    }
                }
            }
            local
        })
        .collect();

    let total_active: usize = active_slabs.iter().map(Vec::len).sum();
    if total_active == 0 {
        return MeshData::default();
    }

    // Step 2: Parallel chunked extraction with sparse edge welding.
    let num_slabs = nx - 1;
    let ranges: Vec<(usize, usize)> = if total_active < 4096 || num_slabs < 2 {
        vec![(0, num_slabs)]
    } else {
        let num_chunks = (rayon::current_num_threads() * 2).min(num_slabs).max(1);
        let target_per_chunk = (total_active / num_chunks).max(1);
        let mut ranges = Vec::with_capacity(num_chunks);
        let mut start = 0;
        let mut acc = 0;
        for (i, slab) in active_slabs.iter().enumerate() {
            acc += slab.len();
            if acc >= target_per_chunk && ranges.len() + 1 < num_chunks && i + 1 < num_slabs {
                ranges.push((start, i + 1));
                start = i + 1;
                acc = 0;
            }
        }
        if start < num_slabs {
            ranges.push((start, num_slabs));
        }
        ranges
    };

    const EDGE_OFFSETS: [(usize, usize, usize, usize); 12] = [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (1, 0, 1, 1),
        (0, 1, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 2),
        (1, 0, 0, 2),
        (1, 1, 0, 2),
        (0, 1, 0, 2),
    ];

    let num_chunks = ranges.len();
    let mut chunks: Vec<ChunkResult> = ranges
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, &(i_start, i_end))| {
            let is_last_chunk = chunk_idx + 1 == num_chunks;
            let chunk_active: usize = active_slabs[i_start..i_end].iter().map(Vec::len).sum();
            let mut edge_map = SparseEdgeMap::with_capacity(chunk_active * 3);
            let mut owned_points = Vec::with_capacity(chunk_active);
            let mut hi_points = Vec::new();
            let mut hi_slots = Vec::new();
            let mut faces = Vec::with_capacity(chunk_active * 2);
            let mut skipped_degenerate = false;

            for slab in &active_slabs[i_start..i_end] {
                for &(i, j, k, cube_index) in slab {
                    let mut em = EDGE_TABLE[cube_index as usize];
                    if em == 0 {
                        continue;
                    }

                    let mut vert_list = [u32::MAX; 12];
                    while em != 0 {
                        let e = em.trailing_zeros() as usize;
                        em &= em - 1;
                        let (di, dj, dk, axis) = EDGE_OFFSETS[e];
                        vert_list[e] = get_or_insert_edge(
                            coords,
                            field,
                            ny,
                            nz,
                            stride_x,
                            i_end,
                            is_last_chunk,
                            &mut edge_map,
                            &mut owned_points,
                            &mut hi_points,
                            &mut hi_slots,
                            i + di,
                            j + dj,
                            k + dk,
                            axis,
                        );
                    }

                    let tri_row = &TRI_TABLE[cube_index as usize];
                    let mut t = 0;
                    while t < 15 && tri_row[t] != -1 {
                        let v0 = vert_list[tri_row[t] as usize];
                        let v1 = vert_list[tri_row[t + 1] as usize];
                        let v2 = vert_list[tri_row[t + 2] as usize];

                        if v0 != v1 && v1 != v2 && v2 != v0 {
                            let p0 = chunk_point(v0, &owned_points, &hi_points);
                            let p1 = chunk_point(v1, &owned_points, &hi_points);
                            let p2 = chunk_point(v2, &owned_points, &hi_points);
                            let e1 = [
                                (p1[0] - p0[0]) as f64,
                                (p1[1] - p0[1]) as f64,
                                (p1[2] - p0[2]) as f64,
                            ];
                            let e2 = [
                                (p2[0] - p0[0]) as f64,
                                (p2[1] - p0[1]) as f64,
                                (p2[2] - p0[2]) as f64,
                            ];
                            let cx = e1[1] * e2[2] - e1[2] * e2[1];
                            let cy = e1[2] * e2[0] - e1[0] * e2[2];
                            let cz = e1[0] * e2[1] - e1[1] * e2[0];
                            let cross_sq = cx * cx + cy * cy + cz * cz;
                            if cross_sq >= 1e-24 {
                                faces.push([v0, v1, v2]);
                            } else {
                                skipped_degenerate = true;
                            }
                        } else {
                            skipped_degenerate = true;
                        }
                        t += 3;
                    }
                }
            }

            ChunkResult {
                edge_map,
                owned_points,
                hi_points,
                hi_slots,
                faces,
                skipped_degenerate,
            }
        })
        .collect();

    if chunks.len() == 1 {
        let chunk = chunks.pop().unwrap();
        let mut mesh = MeshData {
            points: chunk.owned_points,
            faces: chunk.faces,
        };
        if chunk.skipped_degenerate {
            mesh.compact_unreferenced();
        }
        return mesh;
    }

    let mut point_offsets = Vec::with_capacity(chunks.len());
    let mut total_owned_points = 0usize;
    let mut total_faces = 0usize;
    let mut any_skipped_degenerate = false;
    for chunk in &chunks {
        point_offsets.push(total_owned_points as u32);
        total_owned_points += chunk.owned_points.len();
        total_faces += chunk.faces.len();
        any_skipped_degenerate |= chunk.skipped_degenerate;
    }

    let mut extra_points: Vec<[f32; 3]> = Vec::new();
    let mut hi_remaps: Vec<Vec<u32>> = Vec::with_capacity(chunks.len());
    for c in 0..chunks.len() {
        if c + 1 == chunks.len() {
            hi_remaps.push(Vec::new());
            continue;
        }
        let next_map = &chunks[c + 1].edge_map;
        let next_offset = point_offsets[c + 1];
        let mut remap = Vec::with_capacity(chunks[c].hi_slots.len());
        for (hi_idx, &slot) in chunks[c].hi_slots.iter().enumerate() {
            if let Some(next_local_id) = next_map.get(slot) {
                debug_assert_eq!(next_local_id & HI_BIT, 0);
                remap.push(next_offset + next_local_id);
            } else {
                let global_id = (total_owned_points + extra_points.len()) as u32;
                extra_points.push(chunks[c].hi_points[hi_idx]);
                remap.push(global_id);
            }
        }
        hi_remaps.push(remap);
    }

    let total_points = total_owned_points + extra_points.len();
    let mut points = vec![[0.0f32; 3]; total_points];
    let mut faces = vec![[0u32; 3]; total_faces];
    if !extra_points.is_empty() {
        points[total_owned_points..].copy_from_slice(&extra_points);
    }

    let (owned_dst, _) = points.split_at_mut(total_owned_points);
    let point_slices = split_by_lengths(owned_dst, chunks.iter().map(|c| c.owned_points.len()));
    let face_slices = split_by_lengths(&mut faces, chunks.iter().map(|c| c.faces.len()));

    (chunks, point_slices, face_slices, point_offsets, hi_remaps)
        .into_par_iter()
        .for_each(|(chunk, pt_dst, face_dst, pt_off, hi_remap)| {
            pt_dst.copy_from_slice(&chunk.owned_points);
            for (dst, src) in face_dst.iter_mut().zip(chunk.faces.iter()) {
                let mut f = *src;
                for v in &mut f {
                    if (*v & HI_BIT) != 0 {
                        *v = hi_remap[(*v & !HI_BIT) as usize];
                    } else {
                        *v += pt_off;
                    }
                }
                *dst = f;
            }
        });

    let mut mesh = MeshData { points, faces };
    if any_skipped_degenerate {
        mesh.compact_unreferenced();
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold;
    use std::collections::HashMap;

    fn assert_watertight_2_manifold(mesh: &MeshData) {
        assert!(!mesh.points.is_empty());
        assert!(!mesh.faces.is_empty());

        let mut dir_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for f in &mesh.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *dir_counts.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &cnt) in &dir_counts {
            assert_eq!(cnt, 1, "Non-manifold directed edge ({a}, {b}) count {cnt}");
            assert_eq!(
                dir_counts.get(&(b, a)).copied().unwrap_or(0),
                1,
                "Open boundary edge ({a}, {b})"
            );
        }
    }

    #[test]
    fn test_sphere_isosurface_watertight_positive_volume() {
        let n = 25usize;
        let coords: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let mut field = vec![0.0; n * n * n];
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                for (k, &z) in coords.iter().enumerate() {
                    field[(i * n + j) * n + k] = (x * x + y * y + z * z).sqrt() - 1.2;
                }
            }
        }

        let mesh = extract_isosurface_cartesian(&coords, &coords, &coords, &field);
        assert_watertight_2_manifold(&mesh);
        let vol = manifold::signed_volume(&mesh);
        let expected_vol = 4.0 / 3.0 * std::f64::consts::PI * 1.2f64.powi(3);
        assert!(
            vol > 0.0 && (vol - expected_vol).abs() < 0.5,
            "Sphere signed volume {vol} vs expected {expected_vol}"
        );
    }

    #[test]
    fn test_box_clamped_gyroid_watertight() {
        let n = 25usize;
        let coords: Vec<f64> = (0..n)
            .map(|i| -6.0 + 12.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let mut field = vec![1.0; n * n * n];
        let k_wave = 2.0 * std::f64::consts::PI / 5.0;
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                for (k, &z) in coords.iter().enumerate() {
                    if x.abs() <= 4.5 && y.abs() <= 4.5 && z.abs() <= 4.5 {
                        let gx = (k_wave * x).sin() * (k_wave * y).cos()
                            + (k_wave * y).sin() * (k_wave * z).cos()
                            + (k_wave * z).sin() * (k_wave * x).cos();
                        field[(i * n + j) * n + k] = gx.abs() - 0.5;
                    }
                }
            }
        }
        let mesh = extract_isosurface_cartesian(&coords, &coords, &coords, &field);
        assert_watertight_2_manifold(&mesh);
        assert!(manifold::signed_volume(&mesh) > 0.0);
    }

    #[test]
    fn test_parallel_chunked_gyroid_watertight() {
        let n = 48usize;
        let coords: Vec<f64> = (0..n)
            .map(|i| -8.0 + 16.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let mut field = vec![1.0; n * n * n];
        let k_wave = 2.0 * std::f64::consts::PI / 5.0;
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                for (k, &z) in coords.iter().enumerate() {
                    if x.abs() <= 6.5 && y.abs() <= 6.5 && z.abs() <= 6.5 {
                        let gx = (k_wave * x).sin() * (k_wave * y).cos()
                            + (k_wave * y).sin() * (k_wave * z).cos()
                            + (k_wave * z).sin() * (k_wave * x).cos();
                        field[(i * n + j) * n + k] = gx.abs() - 0.5;
                    }
                }
            }
        }
        let mesh = extract_isosurface_cartesian(&coords, &coords, &coords, &field);
        assert_watertight_2_manifold(&mesh);
        assert!(manifold::signed_volume(&mesh) > 0.0);
    }
}
