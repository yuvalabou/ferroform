//! Pure Rust TPMS & volumetric lattice generation engine for PyRoid.

pub mod bvh;
pub mod csg;
pub mod edt;
pub mod manifold;
pub mod marching;
mod mc_tables;
pub mod tpms;

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::marching::{
    extract_isosurface_cartesian, extract_isosurface_cylindrical, MeshData,
};
use crate::tpms::TpmsType;

/// Parameters for radially symmetrical parametric TPMS generation.
#[derive(Clone, Debug)]
pub struct ParametricParams {
    pub tpms_type: TpmsType,
    pub res: usize,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub r1: f64,
    pub r2: f64,
    pub phi_scale: f64,
    pub wall_thickness: f64,
    pub cell_radius: f64,
    pub cell_height: f64,
    pub bias: f64,
    pub include_cavities: bool,
}

impl ParametricParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.res == 0 {
            return Err("Resolution must be a positive integer".to_string());
        }
        if self.a <= 0.0 || self.b <= 0.0 || self.c <= 0.0 {
            return Err("Dimensions (a, b, c) must be positive values".to_string());
        }
        if self.r1 < 0.0 {
            return Err("Inner radius (r1) must be non-negative".to_string());
        }
        if self.wall_thickness <= 0.0 {
            return Err("Wall thickness must be positive".to_string());
        }
        if self.cell_radius <= 0.0 || self.cell_height <= 0.0 {
            return Err("Cell radius and height must be positive".to_string());
        }
        if !(-1.4..=1.4).contains(&self.bias) {
            return Err("Bias must be between -1.4 and 1.4".to_string());
        }
        Ok(())
    }
}

/// Parameters for volumetric STL TPMS lattice infill conversion.
#[derive(Clone, Debug)]
pub struct StlParams {
    pub tpms_type: TpmsType,
    pub resolution: usize,
    pub cell_size: f64,
    pub wall_thickness: f64,
    pub bias: f64,
    pub flat_cap: bool,
    pub preserve_shell: bool,
    pub shell_thickness: f64,
    pub is_skeletal: bool,
    pub infill_percent: Option<f64>,
    pub include_cavities: bool,
}

pub type InfillParams = StlParams;

impl StlParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.resolution <= 10 {
            return Err("Resolution must be at least 10".to_string());
        }
        if self.cell_size <= 0.0 {
            return Err("Cell size must be positive".to_string());
        }
        if self.wall_thickness <= 0.0 {
            return Err("Wall thickness must be positive".to_string());
        }
        if !(-1.4..=1.4).contains(&self.bias) {
            return Err("Bias must be between -1.4 and 1.4".to_string());
        }
        if let Some(pct) = self.infill_percent {
            if !(pct > 0.0 && pct < 100.0) {
                return Err("Infill percent must be between 0 and 100".to_string());
            }
        }
        if self.preserve_shell && self.shell_thickness <= 0.0 {
            return Err(
                "Shell thickness must be positive when shell preservation is enabled".to_string(),
            );
        }
        Ok(())
    }
}

/// Output meshes produced by parametric or STL lattice generation.
#[derive(Clone, Debug)]
pub struct GeneratedLattice {
    pub mesh: MeshData,
    pub cavity_a: Option<MeshData>,
    pub cavity_b: Option<MeshData>,
    pub is_shell_void: bool,
}

/// Double-precision triangle mesh representation for SolidCraft interoperability.
#[derive(Clone, Debug, Default)]
pub struct Trimesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

impl From<&MeshData> for Trimesh {
    fn from(mesh: &MeshData) -> Self {
        Self {
            vertices: mesh
                .points
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect(),
            faces: mesh.faces.clone(),
        }
    }
}

/// Clean Rust trait for external B-Rep / CAD engines (such as SolidCraft).
pub trait LatticeInfillProvider {
    fn generate_infill(&self, boundary: &Trimesh, params: &InfillParams) -> Result<Trimesh, String>;
}

/// Default Rayon-parallel lattice infill provider.
pub struct DefaultLatticeProvider;

impl LatticeInfillProvider for DefaultLatticeProvider {
    fn generate_infill(&self, boundary: &Trimesh, params: &InfillParams) -> Result<Trimesh, String> {
        let out = convert_stl(&boundary.vertices, &boundary.faces, params, |_, _| {})?;
        Ok(Trimesh::from(&out.mesh))
    }
}

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / ((n - 1) as f64);
    (0..n).map(|i| start + (i as f64) * step).collect()
}

/// Generate a radially symmetrical parametric TPMS structure and dual cavities.
pub fn generate_parametric(
    params: &ParametricParams,
    mut progress: impl FnMut(i32, &str),
) -> Result<GeneratedLattice, String> {
    params.validate()?;

    let res = params.res;
    progress(10, &format!("Initializing cylindrical grid (res={res})..."));

    let ky = 2.0 * PI / params.b;
    let r_aux = linspace(0.0, params.a, res);
    let phis = linspace(0.0, params.b, res);
    let zs = linspace(0.0, params.c, res);

    let r_slope = (params.r2 - params.r1) / params.a;
    let rs: Vec<f64> = r_aux.iter().map(|&ra| r_slope * ra + params.r1).collect();
    let r_min = params.r1 - params.wall_thickness;
    let mask_i: Vec<bool> = rs.iter().map(|&r| r >= r_min && r <= params.r1).collect();

    progress(
        35,
        &format!(
            "Evaluating radial {} scalar field...",
            params.tpms_type.display_name()
        ),
    );

    let scale_x = 2.0 * PI * params.cell_radius / params.a;
    let scale_y = 2.0 * PI * params.cell_radius / params.b;
    let scale_z = 2.0 * PI * params.cell_height / params.c;

    let total = res * res * res;
    let mut main_field = vec![1.0; total];
    main_field
        .par_chunks_mut(res * res)
        .enumerate()
        .for_each(|(i, slab)| {
            if !mask_i[i] {
                return;
            }
            let x = r_aux[i] * scale_x;
            for (j, &phi) in phis.iter().enumerate() {
                let y = phi * params.phi_scale * scale_y;
                let row_off = j * res;
                for (k, &z) in zs.iter().enumerate() {
                    slab[row_off + k] =
                        params.tpms_type.eval_radial(x, y, z * scale_z) + params.bias;
                }
            }
        });

    progress(60, "Transforming coordinates to Cartesian...");
    progress(80, "Extracting isosurface contour...");

    let mut mesh = extract_isosurface_cylindrical(&rs, &phis, &zs, ky, &main_field);
    manifold::ensure_positive_volume(&mut mesh);

    let (cavity_a, cavity_b) = if params.include_cavities {
        let t_sheet = 0.2 * (params.tpms_type.thickness_multiplier() / 4.80);
        let mut cav_a_field = vec![1.0; total];
        let mut cav_b_field = vec![1.0; total];

        cav_a_field
            .par_chunks_mut(res * res)
            .zip(cav_b_field.par_chunks_mut(res * res))
            .enumerate()
            .for_each(|(i, (a_slab, b_slab))| {
                if !mask_i[i] {
                    return;
                }
                let m_slab = &main_field[i * res * res..(i + 1) * res * res];
                for idx in 0..res * res {
                    let g = m_slab[idx];
                    a_slab[idx] = t_sheet - g;
                    b_slab[idx] = g + t_sheet;
                }
            });

        let mut cav_a = extract_isosurface_cylindrical(&rs, &phis, &zs, ky, &cav_a_field);
        manifold::ensure_positive_volume(&mut cav_a);
        let mut cav_b = extract_isosurface_cylindrical(&rs, &phis, &zs, ky, &cav_b_field);
        manifold::ensure_positive_volume(&mut cav_b);
        (Some(cav_a), Some(cav_b))
    } else {
        (None, None)
    };

    progress(95, "Finalizing mesh...");

    Ok(GeneratedLattice {
        mesh,
        cavity_a,
        cavity_b,
        is_shell_void: false,
    })
}

/// Convert an input STL triangle mesh volume into a TPMS lattice (or internal void when `preserve_shell=true`).
pub fn convert_stl(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    params: &StlParams,
    mut progress: impl FnMut(i32, &str),
) -> Result<GeneratedLattice, String> {
    params.validate()?;
    if vertices.is_empty() || faces.is_empty() {
        return Err("Input STL mesh is empty".to_string());
    }

    let mut min_b = [f64::INFINITY; 3];
    let mut max_b = [f64::NEG_INFINITY; 3];
    for v in vertices {
        for d in 0..3 {
            min_b[d] = min_b[d].min(v[d]);
            max_b[d] = max_b[d].max(v[d]);
        }
    }
    let extent = [
        max_b[0] - min_b[0],
        max_b[1] - min_b[1],
        max_b[2] - min_b[2],
    ];
    let max_dim = extent[0].max(extent[1]).max(extent[2]);
    if max_dim <= 0.0 {
        return Err("Invalid STL mesh with zero extent".to_string());
    }

    let res = params.resolution as f64;
    let pad = 2.0 * max_dim / res;
    let grid_min = [min_b[0] - pad, min_b[1] - pad, min_b[2] - pad];
    let grid_max = [max_b[0] + pad, max_b[1] + pad, max_b[2] + pad];
    let grid_extent = [
        grid_max[0] - grid_min[0],
        grid_max[1] - grid_min[1],
        grid_max[2] - grid_min[2],
    ];

    let nx = (res * (grid_extent[0] / max_dim)).round().max(12.0) as usize;
    let ny = (res * (grid_extent[1] / max_dim)).round().max(12.0) as usize;
    let nz = (res * (grid_extent[2] / max_dim)).round().max(12.0) as usize;

    progress(10, &format!("Initializing 3D voxel grid ({nx}x{ny}x{nz})..."));

    let xs = linspace(grid_min[0], grid_max[0], nx);
    let ys = linspace(grid_min[1], grid_max[1], ny);
    let zs = linspace(grid_min[2], grid_max[2], nz);
    let total = nx * ny * nz;

    progress(
        25,
        &format!("Testing volume containment ({total} points)..."),
    );

    let inside_3d = bvh::compute_containment_grid(vertices, faces, &xs, &ys, &zs);

    progress(
        50,
        &format!(
            "Evaluating 3D {} scalar field...",
            params.tpms_type.display_name()
        ),
    );

    let is_shell_void = params.preserve_shell && params.shell_thickness > 0.0;
    let k_wave = 2.0 * PI / params.cell_size.max(0.001);
    let mut scalar_val = vec![0.0; total];
    scalar_val
        .par_chunks_mut(ny * nz)
        .enumerate()
        .for_each(|(i, slab)| {
            let kx = k_wave * xs[i];
            let in_slab = &inside_3d[i * ny * nz..(i + 1) * ny * nz];
            for (j, &y) in ys.iter().enumerate() {
                let ky = k_wave * y;
                let row_off = j * nz;
                for (k, &z) in zs.iter().enumerate() {
                    if is_shell_void || in_slab[row_off + k] {
                        slab[row_off + k] =
                            params.tpms_type.eval(kx, ky, k_wave * z) + params.bias;
                    }
                }
            }
        });

    let any_inside = inside_3d.iter().any(|&b| b);
    let q = params
        .infill_percent
        .filter(|_| any_inside)
        .map(|pct| (pct / 100.0).clamp(0.01, 0.99));

    let t_norm = params.tpms_type.thickness_multiplier() / 4.80;
    let t = if let (Some(qv), false) = (q, params.is_skeletal) {
        bvh::compute_abs_quantile(&scalar_val, &inside_3d, qv)
    } else {
        (params.tpms_type.thickness_multiplier() * params.wall_thickness
            / params.cell_size.max(0.001))
        .clamp(0.05, 1.5 * t_norm)
    };

    let skeletal_thresh = if params.is_skeletal {
        let r = params.wall_thickness / params.cell_size.max(0.001);
        let scale_strut = 3.0 * t_norm;
        if let Some(qv) = q {
            bvh::compute_quantile(&scalar_val, &inside_3d, qv)
        } else {
            (scale_strut * (r - 0.2)).clamp(-1.1 * t_norm, 1.1 * t_norm)
        }
    } else {
        0.0
    };

    let extract_cavities = |mask: &[bool]| -> (Option<MeshData>, Option<MeshData>) {
        if !params.include_cavities || params.is_skeletal {
            return (None, None);
        }
        let field_a: Vec<f64> = scalar_val
            .par_iter()
            .zip(mask.par_iter())
            .map(|(&s, &m)| if m { t - s } else { 1.0 })
            .collect();
        let field_b: Vec<f64> = scalar_val
            .par_iter()
            .zip(mask.par_iter())
            .map(|(&s, &m)| if m { s + t } else { 1.0 })
            .collect();
        let mut cav_a = extract_isosurface_cartesian(&xs, &ys, &zs, &field_a);
        manifold::ensure_positive_volume(&mut cav_a);
        let mut cav_b = extract_isosurface_cartesian(&xs, &ys, &zs, &field_b);
        manifold::ensure_positive_volume(&mut cav_b);
        (Some(cav_a), Some(cav_b))
    };

    if is_shell_void {
        progress(65, "Computing shell distance transform...");
        let dx = grid_extent[0] / ((nx - 1).max(1) as f64);
        let dy = grid_extent[1] / ((ny - 1).max(1) as f64);
        let dz = grid_extent[2] / ((nz - 1).max(1) as f64);

        let dist_inside = edt::edt_3d(&inside_3d, nx, ny, nz, dx, dy, dz);
        let outside_3d: Vec<bool> = inside_3d.par_iter().map(|&b| !b).collect();
        let dist_outside = edt::edt_3d(&outside_3d, nx, ny, nz, dx, dy, dz);

        let inner_mask: Vec<bool> = dist_inside
            .par_iter()
            .map(|&d| d > params.shell_thickness)
            .collect();
        let field_void: Vec<f64> = dist_inside
            .par_iter()
            .zip(dist_outside.par_iter())
            .zip(scalar_val.par_iter())
            .map(|((&d_in, &d_out), &s)| {
                if d_in <= 0.0 {
                    params.shell_thickness + d_out
                } else {
                    let vs = if params.is_skeletal {
                        skeletal_thresh - s
                    } else {
                        t - s.abs()
                    };
                    (params.shell_thickness - d_in).max(vs)
                }
            })
            .collect();

        progress(75, "Extracting internal void isosurface...");
        let mut void_mesh = extract_isosurface_cartesian(&xs, &ys, &zs, &field_void);
        manifold::ensure_positive_volume(&mut void_mesh);

        progress(85, "Performing manifold shell boolean subtraction...");
        let outer_mesh = MeshData {
            points: vertices
                .iter()
                .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                .collect(),
            faces: faces.to_vec(),
        };
        let mut shell_mesh = csg::subtract_interior_void(&outer_mesh, &void_mesh);
        if params.flat_cap {
            progress(90, "Applying flat Z-plane caps...");
            shell_mesh = csg::trim_flat_cap(&shell_mesh);
        }

        let (cavity_a, cavity_b) = extract_cavities(&inner_mask);
        progress(95, "Finalizing mesh...");
        return Ok(GeneratedLattice {
            mesh: shell_mesh,
            cavity_a,
            cavity_b,
            is_shell_void: false,
        });
    }

    progress(75, "Extracting gyroid isosurface contour...");
    let field: Vec<f64> = scalar_val
        .par_iter()
        .zip(inside_3d.par_iter())
        .map(|(&s, &is_in)| {
            if !is_in {
                1.0
            } else if params.is_skeletal {
                s - skeletal_thresh
            } else {
                s.abs() - t
            }
        })
        .collect();

    let mut mesh = extract_isosurface_cartesian(&xs, &ys, &zs, &field);
    manifold::ensure_positive_volume(&mut mesh);
    if params.flat_cap {
        progress(88, "Applying flat Z-plane caps...");
        mesh = csg::trim_flat_cap(&mesh);
    }
    let (cavity_a, cavity_b) = extract_cavities(&inside_3d);

    progress(95, "Finalizing mesh...");

    Ok(GeneratedLattice {
        mesh,
        cavity_a,
        cavity_b,
        is_shell_void: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_trimesh(half: f64) -> Trimesh {
        let vertices = vec![
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
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
        Trimesh { vertices, faces }
    }

    #[test]
    fn test_parametric_generation_all_tpms() {
        let tpms_list = [
            TpmsType::Gyroid,
            TpmsType::Diamond,
            TpmsType::Primitive,
            TpmsType::IWP,
            TpmsType::FischerKochS,
            TpmsType::Neovius,
        ];
        for tpms_type in tpms_list {
            let params = ParametricParams {
                tpms_type,
                res: 24,
                a: 24.0,
                b: 24.0,
                c: 10.0,
                r1: 12.0,
                r2: 0.0,
                phi_scale: 8.0,
                wall_thickness: 11.5,
                cell_radius: 2.0,
                cell_height: 3.0,
                bias: 0.0,
                include_cavities: true,
            };
            let mut updates = Vec::new();
            let out = generate_parametric(&params, |pct, msg| {
                updates.push((pct, msg.to_string()));
            })
            .unwrap();
            assert!(out.mesh.n_points() > 0);
            assert!(out.mesh.n_cells() > 0);
            assert!(out.cavity_a.as_ref().unwrap().n_cells() > 0);
            assert!(out.cavity_b.as_ref().unwrap().n_cells() > 0);
            assert!(updates.len() >= 4);
        }
    }

    #[test]
    fn test_stl_conversion_and_shell_void_watertight() {
        let cube = box_trimesh(6.0);
        let params = StlParams {
            tpms_type: TpmsType::Gyroid,
            resolution: 30,
            cell_size: 4.0,
            wall_thickness: 0.6,
            bias: 0.0,
            flat_cap: false,
            preserve_shell: true,
            shell_thickness: 1.0,
            is_skeletal: false,
            infill_percent: None,
            include_cavities: true,
        };
        let out = convert_stl(&cube.vertices, &cube.faces, &params, |_, _| {}).unwrap();
        assert!(!out.is_shell_void);
        assert!(out.mesh.n_cells() > 12);
        let vol = manifold::signed_volume(&out.mesh);
        assert!(vol > 0.0 && vol < 12.0 * 12.0 * 12.0);

        let provider = DefaultLatticeProvider;
        let inf_params = StlParams {
            preserve_shell: false,
            ..params
        };
        let tri = provider.generate_infill(&cube, &inf_params).unwrap();
        assert!(!tri.vertices.is_empty());
        assert!(!tri.faces.is_empty());
    }
}
