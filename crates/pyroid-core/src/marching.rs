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
        phis: &'a [f64],
        zs: &'a [f64],
        ky: f64,
    },
}

impl<'a> GridCoords<'a> {
    #[inline]
    fn dims(&self) -> (usize, usize, usize) {
        match self {
            Self::Cartesian { xs, ys, zs } => (xs.len(), ys.len(), zs.len()),
            Self::Cylindrical { rs, phis, zs, .. } => (rs.len(), phis.len(), zs.len()),
        }
    }

    #[inline]
    fn pos(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        match self {
            Self::Cartesian { xs, ys, zs } => [xs[i], ys[j], zs[k]],
            Self::Cylindrical { rs, phis, zs, ky } => {
                let r = rs[i];
                let angle = phis[j] * ky;
                [r * angle.cos(), r * angle.sin(), zs[k]]
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
    extract_isosurface(&GridCoords::Cylindrical { rs, phis, zs, ky }, field)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn get_or_insert_edge(
    coords: &GridCoords<'_>,
    field: &[f64],
    ny: usize,
    nz: usize,
    stride_x: usize,
    edge_verts: &mut [u32],
    points: &mut Vec<[f32; 3]>,
    gi: usize,
    gj: usize,
    gk: usize,
    axis: usize,
) -> u32 {
    let slot_idx = ((gi * ny + gj) * nz + gk) * 3 + axis;
    let existing = edge_verts[slot_idx];
    if existing != u32::MAX {
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

    let vert_id = points.len() as u32;
    points.push(pt);
    edge_verts[slot_idx] = vert_id;
    vert_id
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

    // Step 2: Weld edge vertices using flat O(1) lookup table (nx * ny * nz * 3).
    let mut edge_verts = vec![u32::MAX; nx * ny * nz * 3];
    let mut points = Vec::with_capacity(total_active);
    let mut faces = Vec::with_capacity(total_active * 2);
    let mut skipped_degenerate = false;

    for slab in active_slabs {
        for (i, j, k, cube_index) in slab {
            let em = EDGE_TABLE[cube_index as usize];
            if em == 0 {
                continue;
            }

            let mut vert_list = [u32::MAX; 12];
            if (em & (1 << 0)) != 0 {
                vert_list[0] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j, k, 0);
            }
            if (em & (1 << 1)) != 0 {
                vert_list[1] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i + 1, j, k, 1);
            }
            if (em & (1 << 2)) != 0 {
                vert_list[2] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j + 1, k, 0);
            }
            if (em & (1 << 3)) != 0 {
                vert_list[3] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j, k, 1);
            }
            if (em & (1 << 4)) != 0 {
                vert_list[4] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j, k + 1, 0);
            }
            if (em & (1 << 5)) != 0 {
                vert_list[5] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i + 1, j, k + 1, 1);
            }
            if (em & (1 << 6)) != 0 {
                vert_list[6] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j + 1, k + 1, 0);
            }
            if (em & (1 << 7)) != 0 {
                vert_list[7] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j, k + 1, 1);
            }
            if (em & (1 << 8)) != 0 {
                vert_list[8] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j, k, 2);
            }
            if (em & (1 << 9)) != 0 {
                vert_list[9] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i + 1, j, k, 2);
            }
            if (em & (1 << 10)) != 0 {
                vert_list[10] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i + 1, j + 1, k, 2);
            }
            if (em & (1 << 11)) != 0 {
                vert_list[11] = get_or_insert_edge(coords, field, ny, nz, stride_x, &mut edge_verts, &mut points, i, j + 1, k, 2);
            }

            let tri_row = &TRI_TABLE[cube_index as usize];
            let mut t = 0;
            while t < 15 && tri_row[t] != -1 {
                let v0 = vert_list[tri_row[t] as usize];
                let v1 = vert_list[tri_row[t + 1] as usize];
                let v2 = vert_list[tri_row[t + 2] as usize];

                if v0 != v1 && v1 != v2 && v2 != v0 {
                    let p0 = points[v0 as usize];
                    let p1 = points[v1 as usize];
                    let p2 = points[v2 as usize];
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

    let mut mesh = MeshData { points, faces };
    if skipped_degenerate {
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
}
