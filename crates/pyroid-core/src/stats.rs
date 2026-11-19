//! Quantitative mesh statistics, infill/grading solvers, and per-vertex density scalars.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::grading::GradingMode;
use crate::manifold::signed_volume;
use crate::marching::MeshData;
use crate::tpms::TpmsType;
use crate::StlParams;

/// Quantitative mesh and manufacturability metrics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MeshStats {
    pub n_points: usize,
    pub n_cells: usize,
    pub volume: f64,
    pub surface_area: f64,
    pub porosity_pct: Option<f64>,
    pub sa_to_vol_ratio: f64,
    pub n_open_edges: usize,
    pub is_watertight: bool,
}

/// Compute surface area, signed volume, open edge count, and porosity for a mesh.
pub fn compute_mesh_stats(mesh: &MeshData, reference_volume: Option<f64>) -> MeshStats {
    let n_points = mesh.points.len();
    let n_cells = mesh.faces.len();

    let mut surface_area = 0.0f64;
    let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::with_capacity(n_cells * 3 / 2);

    for f in &mesh.faces {
        let p0 = mesh.points[f[0] as usize];
        let p1 = mesh.points[f[1] as usize];
        let p2 = mesh.points[f[2] as usize];

        let ax = (p1[0] - p0[0]) as f64;
        let ay = (p1[1] - p0[1]) as f64;
        let az = (p1[2] - p0[2]) as f64;
        let bx = (p2[0] - p0[0]) as f64;
        let by = (p2[1] - p0[1]) as f64;
        let bz = (p2[2] - p0[2]) as f64;

        let cx = ay * bz - az * by;
        let cy = az * bx - ax * bz;
        let cz = ax * by - ay * bx;
        surface_area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();

        let edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])];
        for (u, v) in edges {
            let key = if u < v { (u, v) } else { (v, u) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }

    let n_open_edges = edge_counts.values().filter(|&&c| c != 2).count();
    let volume = signed_volume(mesh).abs();
    let sa_to_vol_ratio = if volume > 0.0 {
        surface_area / volume
    } else {
        0.0
    };
    let is_watertight = n_open_edges == 0 && n_cells > 0;

    let porosity_pct = reference_volume.and_then(|rv| {
        if rv > 0.0 {
            Some(((1.0 - (volume / rv)) * 100.0).clamp(0.0, 100.0))
        } else {
            None
        }
    });

    MeshStats {
        n_points,
        n_cells,
        volume,
        surface_area,
        porosity_pct,
        sa_to_vol_ratio,
        n_open_edges,
        is_watertight,
    }
}

/// Estimate solid infill percentage (%) from cell size, wall thickness, and TPMS geometry.
pub fn estimate_infill(
    cell_size: f64,
    wall_thickness: f64,
    is_skeletal: bool,
    tpms_type: TpmsType,
) -> f64 {
    if cell_size <= 0.0 {
        return 30.0;
    }
    let ratio = wall_thickness / cell_size;
    let raw = if is_skeletal {
        50.0 + 95.5 * (ratio - 0.2)
    } else {
        tpms_type.surface_factor() * ratio
    };
    raw.clamp(1.0, 99.0)
}

/// Solve unit cell size (mm) from target infill % and wall thickness.
pub fn solve_cell_size(
    infill_pct: f64,
    wall_thickness: f64,
    is_skeletal: bool,
    tpms_type: TpmsType,
) -> f64 {
    let pct = infill_pct.clamp(1.0, 99.0);
    let ratio = if is_skeletal {
        ((pct - 50.0) / 95.5 + 0.2).max(0.05)
    } else {
        (pct / tpms_type.surface_factor()).max(0.02)
    };
    (wall_thickness / ratio).max(0.1)
}

/// Solve wall thickness (mm) from target infill % and unit cell size.
pub fn solve_wall_thickness(
    infill_pct: f64,
    cell_size: f64,
    is_skeletal: bool,
    tpms_type: TpmsType,
) -> f64 {
    let pct = infill_pct.clamp(1.0, 99.0);
    let ratio = if is_skeletal {
        ((pct - 50.0) / 95.5 + 0.2).max(0.05)
    } else {
        (pct / tpms_type.surface_factor()).max(0.02)
    };
    (cell_size * ratio).max(0.05)
}

/// Compute per-vertex estimated infill density (%) for 3D viewport coloring.
pub fn compute_density_scalars(
    mesh: &MeshData,
    stl_params: Option<&StlParams>,
    input_bounds: Option<([f64; 3], [f64; 3])>,
    surface_points: Option<&[[f32; 3]]>,
    porosity_pct: Option<f64>,
) -> Vec<f32> {
    let n_pts = mesh.points.len();
    if n_pts == 0 {
        return Vec::new();
    }

    let Some(params) = stl_params else {
        let val = porosity_pct
            .map(|p| (100.0 - p).clamp(0.0, 100.0) as f32)
            .unwrap_or(50.0);
        return vec![val; n_pts];
    };

    if params.grading.mode == GradingMode::None {
        let val = estimate_infill(
            params.cell_size,
            params.wall_thickness,
            params.is_skeletal,
            params.tpms_type,
        ) as f32;
        return vec![val; n_pts];
    }

    let (min_b, max_b) = input_bounds.unwrap_or_else(|| mesh_bounds(&mesh.points));
    let ex = (max_b[0] - min_b[0]).max(1e-6);
    let ey = (max_b[1] - min_b[1]).max(1e-6);
    let ez = (max_b[2] - min_b[2]).max(1e-6);

    let u_vals: Vec<f64> = match params.grading.mode {
        GradingMode::AxialX => mesh
            .points
            .par_iter()
            .map(|p| ((p[0] as f64 - min_b[0]) / ex).clamp(0.0, 1.0))
            .collect(),
        GradingMode::AxialY => mesh
            .points
            .par_iter()
            .map(|p| ((p[1] as f64 - min_b[1]) / ey).clamp(0.0, 1.0))
            .collect(),
        GradingMode::AxialZ => mesh
            .points
            .par_iter()
            .map(|p| ((p[2] as f64 - min_b[2]) / ez).clamp(0.0, 1.0))
            .collect(),
        GradingMode::Radial => {
            let cx = min_b[0] + 0.5 * ex;
            let cy = min_b[1] + 0.5 * ey;
            let max_r = (0.5 * ex.max(ey)).max(1e-6);
            mesh.points
                .par_iter()
                .map(|p| ((p[0] as f64 - cx).hypot(p[1] as f64 - cy) / max_r).clamp(0.0, 1.0))
                .collect()
        }
        GradingMode::ShellToCore => {
            if let Some(surf) = surface_points.filter(|s| !s.is_empty()) {
                let dists: Vec<f64> = mesh
                    .points
                    .par_iter()
                    .map(|p| {
                        let mut best_sq = f32::INFINITY;
                        for sp in surf {
                            let dx = p[0] - sp[0];
                            let dy = p[1] - sp[1];
                            let dz = p[2] - sp[2];
                            let sq = dx * dx + dy * dy + dz * dz;
                            if sq < best_sq {
                                best_sq = sq;
                            }
                        }
                        (best_sq as f64).sqrt()
                    })
                    .collect();
                let depth = params
                    .grading
                    .transition_depth
                    .filter(|&d| d > 0.0)
                    .unwrap_or_else(|| dists.iter().copied().fold(0.0f64, f64::max).max(1e-6));
                dists
                    .into_iter()
                    .map(|d| (d / depth).clamp(0.0, 1.0))
                    .collect()
            } else {
                vec![0.0; n_pts]
            }
        }
        GradingMode::None => vec![0.0; n_pts],
    };

    let w_start = params.wall_thickness;
    let w_end = params.grading.wall_thickness_end.unwrap_or(w_start);
    let c_start = params.cell_size;
    let c_end = params.grading.cell_size_end.unwrap_or(c_start);
    let factor = params.tpms_type.surface_factor();

    u_vals
        .into_iter()
        .map(|u| {
            let w_u = w_start + (w_end - w_start) * u;
            let c_u = (c_start + (c_end - c_start) * u).max(1e-6);
            let ratio = w_u / c_u;
            let density = if params.is_skeletal {
                (50.0 + 95.5 * (ratio - 0.2)).clamp(5.0, 95.0)
            } else {
                (factor * ratio).clamp(5.0, 95.0)
            };
            density as f32
        })
        .collect()
}

/// Compute axis-aligned bounding box `([min_x, min_y, min_z], [max_x, max_y, max_z])`.
pub fn mesh_bounds(points: &[[f32; 3]]) -> ([f64; 3], [f64; 3]) {
    if points.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut min_b = [f64::INFINITY; 3];
    let mut max_b = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 {
            let v = p[i] as f64;
            if v < min_b[i] {
                min_b[i] = v;
            }
            if v > max_b[i] {
                max_b[i] = v;
            }
        }
    }
    (min_b, max_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infill_and_solvers_round_trip() {
        let cell = solve_cell_size(30.0, 0.8, false, TpmsType::Gyroid);
        let est = estimate_infill(cell, 0.8, false, TpmsType::Gyroid);
        assert!((est - 30.0).abs() < 0.1);

        let wall = solve_wall_thickness(40.0, 5.0, false, TpmsType::Diamond);
        let est2 = estimate_infill(5.0, wall, false, TpmsType::Diamond);
        assert!((est2 - 40.0).abs() < 0.1);
    }

    #[test]
    fn stats_on_unit_cube() {
        let mesh = MeshData {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            faces: vec![
                [0, 2, 1],
                [0, 3, 2], // bottom
                [4, 5, 6],
                [4, 6, 7], // top
                [0, 1, 5],
                [0, 5, 4], // front
                [1, 2, 6],
                [1, 6, 5], // right
                [2, 3, 7],
                [2, 7, 6], // back
                [3, 0, 4],
                [3, 4, 7], // left
            ],
        };
        let s = compute_mesh_stats(&mesh, Some(2.0));
        assert!(s.is_watertight);
        assert_eq!(s.n_open_edges, 0);
        assert!((s.volume - 1.0).abs() < 1e-6);
        assert!((s.surface_area - 6.0).abs() < 1e-6);
        assert!((s.porosity_pct.unwrap() - 50.0).abs() < 1e-6);
    }
}
