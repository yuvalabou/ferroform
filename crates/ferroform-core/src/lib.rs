//! Pure Rust TPMS & volumetric lattice generation engine for FerroForm.

pub mod bvh;
pub mod csg;
pub mod edt;
pub mod grading;
pub mod io;
pub mod manifold;
pub mod marching;
mod mc_tables;
pub mod stats;
pub mod tpms;
pub mod voronoi;

use std::f64::consts::PI;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::grading::{
    compute_grading_factors, eval_tpms_with_grading, lerp, GradingMode, GradingParams,
};
use crate::marching::{
    extract_isosurface_cartesian, extract_isosurface_cylindrical, MeshData,
};
use crate::tpms::TpmsType;
use crate::voronoi::{effective_min_dist, smooth_scalar_field, VoronoiField};

fn default_seed() -> u64 {
    42
}

/// Domain shape for parametric TPMS generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParametricShape {
    #[default]
    Cylinder,
    Box,
}

impl ParametricShape {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "cylinder" | "circle" | "radial" => Ok(Self::Cylinder),
            "box" | "cube" | "cartesian" => Ok(Self::Box),
            other => Err(format!(
                "Unknown parametric shape: '{other}'. Expected 'cylinder' or 'box'"
            )),
        }
    }
}

/// Parameters for parametric TPMS generation (cylindrical or Cartesian box).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ParametricParams {
    pub tpms_type: TpmsType,
    pub shape: ParametricShape,
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
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub blend_passes: usize,
}

impl Default for ParametricParams {
    fn default() -> Self {
        Self {
            tpms_type: TpmsType::Gyroid,
            shape: ParametricShape::Cylinder,
            res: 80,
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
            seed: default_seed(),
            blend_passes: 0,
        }
    }
}

impl ParametricParams {
    /// Return named parametric presets matching the legacy Python preset definitions.
    pub fn presets() -> [(&'static str, Self); 4] {
        [
            ("default", Self::default()),
            (
                "fine_detail",
                Self {
                    res: 120,
                    cell_radius: 1.5,
                    cell_height: 2.5,
                    ..Self::default()
                },
            ),
            (
                "thick_walls",
                Self {
                    wall_thickness: 15.0,
                    ..Self::default()
                },
            ),
            (
                "dense_pattern",
                Self {
                    res: 100,
                    phi_scale: 12.0,
                    cell_radius: 1.2,
                    cell_height: 1.8,
                    ..Self::default()
                },
            ),
        ]
    }

    /// Reference bounding volume used for porosity calculation.
    pub fn reference_volume(&self) -> f64 {
        match self.shape {
            ParametricShape::Box => self.a * self.b * self.c,
            ParametricShape::Cylinder => {
                let r_inner = (self.r1 - self.wall_thickness).max(0.0);
                let r_outer = self.r1;
                PI * (r_outer * r_outer - r_inner * r_inner).max(0.0) * self.c
            }
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.res == 0 {
            return Err("Resolution must be a positive integer".to_string());
        }
        if self.a <= 0.0 || self.b <= 0.0 || self.c <= 0.0 {
            return Err("Dimensions (a, b, c) must be positive values".to_string());
        }
        if self.shape == ParametricShape::Cylinder {
            if self.r1 < 0.0 {
                return Err("Inner radius (r1) must be non-negative".to_string());
            }
            if self.wall_thickness <= 0.0 {
                return Err("Wall thickness must be positive".to_string());
            }
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    pub grading: GradingParams,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub blend_passes: usize,
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
        self.grading.validate()?;
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

/// Generate a parametric TPMS structure (cylindrical or Cartesian box) and dual cavities.
pub fn generate_parametric(
    params: &ParametricParams,
    mut progress: impl FnMut(i32, &str),
) -> Result<GeneratedLattice, String> {
    params.validate()?;

    let res = params.res;
    let total = res * res * res;
    let mut main_field = vec![1.0; total];

    let scale_x = 2.0 * PI * params.cell_radius / params.a;
    let scale_y = 2.0 * PI * params.cell_radius / params.b;
    let scale_z = 2.0 * PI * params.cell_height / params.c;

    let zs = linspace(0.0, params.c, res);
    let voronoi_field = if params.tpms_type == TpmsType::Voronoi {
        let v_cell = (params.a / params.cell_radius.max(0.1)).max(0.5);
        let pad = 1.5 * v_cell;
        let (min_b, max_b) = match params.shape {
            ParametricShape::Cylinder => (
                [-params.r1 - pad, -params.r1 - pad, -pad],
                [params.r1 + pad, params.r1 + pad, params.c + pad],
            ),
            ParametricShape::Box => (
                [-0.5 * params.a - pad, -0.5 * params.b - pad, -pad],
                [0.5 * params.a + pad, 0.5 * params.b + pad, params.c + pad],
            ),
        };
        Some((VoronoiField::new(min_b, max_b, v_cell, params.seed), v_cell))
    } else {
        None
    };

    // Coordinate grid and field evaluation differ by shape; cavity extraction shares the rest.
    let (mask_i, jk_bounds, xs_or_rs, ys_or_phis, cyl_ky) = match params.shape {
        ParametricShape::Cylinder => {
            progress(10, &format!("Initializing cylindrical grid (res={res})..."));
            let ky = 2.0 * PI / params.b;
            let r_aux = linspace(0.0, params.a, res);
            let phis = linspace(0.0, params.b, res);

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

            main_field
                .par_chunks_mut(res * res)
                .enumerate()
                .for_each(|(i, slab)| {
                    if !mask_i[i] {
                        return;
                    }
                    let x = r_aux[i] * scale_x;
                    let r_phys = rs[i];
                    for (j, &phi) in phis.iter().enumerate() {
                        let y = phi * params.phi_scale * scale_y;
                        let angle = ky * phi;
                        let px = r_phys * angle.cos();
                        let py = r_phys * angle.sin();
                        let row_off = j * res;
                        for (k, &z) in zs.iter().enumerate() {
                            slab[row_off + k] = if let Some((ref vf, v_cell)) = voronoi_field {
                                vf.eval_strut(px, py, z) - 0.35 * v_cell + params.bias
                            } else {
                                params.tpms_type.eval_radial(x, y, z * scale_z) + params.bias
                            };
                        }
                    }
                });

            progress(60, "Transforming coordinates to Cartesian...");
            (mask_i, (0..res, 0..res), rs, phis, Some(ky))
        }
        ParametricShape::Box => {
            progress(10, &format!("Initializing box grid (res={res})..."));
            let xs = linspace(-params.a * 0.5, params.a * 0.5, res);
            let ys = linspace(-params.b * 0.5, params.b * 0.5, res);
            let mask_i: Vec<bool> = (0..res).map(|i| i > 0 && i + 1 < res).collect();

            progress(
                35,
                &format!(
                    "Evaluating Cartesian {} scalar field...",
                    params.tpms_type.display_name()
                ),
            );

            let inner_end = res.saturating_sub(1);
            main_field
                .par_chunks_mut(res * res)
                .enumerate()
                .for_each(|(i, slab)| {
                    if !mask_i[i] {
                        return;
                    }
                    let x = xs[i] * scale_x;
                    let px = xs[i];
                    for j in 1..inner_end {
                        let y = ys[j] * scale_y;
                        let py = ys[j];
                        let row_off = j * res;
                        for k in 1..inner_end {
                            slab[row_off + k] = if let Some((ref vf, v_cell)) = voronoi_field {
                                vf.eval_strut(px, py, zs[k]) - 0.35 * v_cell + params.bias
                            } else {
                                params.tpms_type.eval(x, y, zs[k] * scale_z) + params.bias
                            };
                        }
                    }
                });

            progress(60, "Preparing Cartesian box domain...");
            (mask_i, (1..inner_end, 1..inner_end), xs, ys, None)
        }
    };

    if params.blend_passes > 0 {
        smooth_scalar_field(&mut main_field, res, res, res, params.blend_passes);
    }

    let extract = |field: &[f64]| -> MeshData {
        let mut m = match cyl_ky {
            Some(ky) => extract_isosurface_cylindrical(&xs_or_rs, &ys_or_phis, &zs, ky, field),
            None => extract_isosurface_cartesian(&xs_or_rs, &ys_or_phis, &zs, field),
        };
        manifold::ensure_positive_volume(&mut m);
        m
    };

    progress(80, "Extracting isosurface contour...");
    let mesh = extract(&main_field);

    let (cavity_a, cavity_b) = if params.include_cavities && params.tpms_type != TpmsType::Voronoi {
        let t_sheet = 0.2 * (params.tpms_type.thickness_multiplier() / 4.80);
        let mut cav_a_field = vec![1.0; total];
        let mut cav_b_field = vec![1.0; total];
        let (j_range, k_range) = jk_bounds;

        cav_a_field
            .par_chunks_mut(res * res)
            .zip(cav_b_field.par_chunks_mut(res * res))
            .enumerate()
            .for_each(|(i, (a_slab, b_slab))| {
                if !mask_i[i] {
                    return;
                }
                let m_slab = &main_field[i * res * res..(i + 1) * res * res];
                for j in j_range.clone() {
                    let row_off = j * res;
                    for k in k_range.clone() {
                        let idx = row_off + k;
                        let g = m_slab[idx];
                        a_slab[idx] = t_sheet - g;
                        b_slab[idx] = g + t_sheet;
                    }
                }
            });

        (Some(extract(&cav_a_field)), Some(extract(&cav_b_field)))
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

    let is_shell_void = params.preserve_shell && params.shell_thickness > 0.0;
    let grading_active = params.grading.is_active();

    let dx = grid_extent[0] / ((nx - 1).max(1) as f64);
    let dy = grid_extent[1] / ((ny - 1).max(1) as f64);
    let dz = grid_extent[2] / ((nz - 1).max(1) as f64);

    progress(
        38,
        if is_shell_void {
            "Computing shell distance transform..."
        } else {
            "Computing signed distance field..."
        },
    );
    let mut dist_inside = Some(edt::edt_3d(&inside_3d, nx, ny, nz, dx, dy, dz));

    let u_field = if grading_active {
        Some(compute_grading_factors(
            &params.grading,
            &xs,
            &ys,
            &zs,
            min_b,
            extent,
            dist_inside.as_deref(),
        ))
    } else {
        None
    };

    progress(
        50,
        &format!(
            "Evaluating 3D {} scalar field...",
            params.tpms_type.display_name()
        ),
    );

    let center = [
        min_b[0] + 0.5 * extent[0],
        min_b[1] + 0.5 * extent[1],
        min_b[2] + 0.5 * extent[2],
    ];
    let k_wave = 2.0 * PI / params.cell_size.max(0.001);
    let mut scalar_val = vec![0.0; total];
    let is_voronoi = params.tpms_type == TpmsType::Voronoi;

    if is_voronoi {
        let r_min = effective_min_dist(params.cell_size, params.wall_thickness, params.is_skeletal);
        let pad = 1.5 * r_min;
        let v_min = [
            xs[0] - pad,
            ys[0] - pad,
            zs[0] - pad,
        ];
        let v_max = [
            xs[nx - 1] + pad,
            ys[ny - 1] + pad,
            zs[nz - 1] + pad,
        ];
        let vfield = VoronoiField::new(v_min, v_max, r_min, params.seed);
        scalar_val
            .par_chunks_mut(ny * nz)
            .enumerate()
            .for_each(|(i, slab)| {
                let x = xs[i];
                for (j, &y) in ys.iter().enumerate() {
                    let row_off = j * nz;
                    for (k, &z) in zs.iter().enumerate() {
                        slab[row_off + k] = if params.is_skeletal {
                            vfield.eval_strut(x, y, z)
                        } else {
                            vfield.eval_foam(x, y, z)
                        };
                    }
                }
            });
    } else {
        scalar_val
            .par_chunks_mut(ny * nz)
            .enumerate()
            .for_each(|(i, slab)| {
                let x = xs[i];
                let kx = k_wave * x;
                let u_slab = u_field
                    .as_ref()
                    .map(|uf| &uf[i * ny * nz..(i + 1) * ny * nz]);
                for (j, &y) in ys.iter().enumerate() {
                    let ky = k_wave * y;
                    let row_off = j * nz;
                    for (k, &z) in zs.iter().enumerate() {
                        let idx = row_off + k;
                        slab[idx] = if let Some(us) = u_slab {
                            let u = us[idx];
                            let bias = params
                                .grading
                                .bias_end
                                .map_or(params.bias, |b_end| lerp(params.bias, b_end, u));
                            eval_tpms_with_grading(
                                params.tpms_type,
                                x,
                                y,
                                z,
                                u,
                                params.cell_size,
                                params.grading.cell_size_end,
                                params.grading.mode,
                                min_b,
                                extent,
                                center,
                            ) + bias
                        } else {
                            params.tpms_type.eval(kx, ky, k_wave * z) + params.bias
                        };
                    }
                }
            });
    }

    if params.blend_passes > 0 {
        progress(58, "Applying voxel scalar-field blending pass...");
        smooth_scalar_field(&mut scalar_val, nx, ny, nz, params.blend_passes);
    }

    let any_inside = inside_3d.iter().any(|&b| b);
    let q = params
        .infill_percent
        .filter(|_| any_inside)
        .map(|pct| (pct / 100.0).clamp(0.01, 0.99));

    let t_norm = params.tpms_type.thickness_multiplier() / 4.80;
    let has_graded_thresh = grading_active
        && (params.grading.wall_thickness_end.is_some() || params.grading.cell_size_end.is_some());

    let sheet_t_at = |u: f64| -> f64 {
        let w = params
            .grading
            .wall_thickness_end
            .map_or(params.wall_thickness, |w_end| {
                lerp(params.wall_thickness, w_end, u)
            });
        if is_voronoi {
            return w.max(0.05);
        }
        let c = params
            .grading
            .cell_size_end
            .map_or(params.cell_size, |c_end| lerp(params.cell_size, c_end, u));
        (params.tpms_type.thickness_multiplier() * w / c.max(0.001)).clamp(0.05, 1.5 * t_norm)
    };

    let skel_t_at = |u: f64| -> f64 {
        let w = params
            .grading
            .wall_thickness_end
            .map_or(params.wall_thickness, |w_end| {
                lerp(params.wall_thickness, w_end, u)
            });
        if is_voronoi {
            // d3 - d1 ~= sqrt(3) * r_strut, where r_strut = 0.5 * wall_thickness
            return 0.5 * w.max(0.05) * 3.0_f64.sqrt();
        }
        let c = params
            .grading
            .cell_size_end
            .map_or(params.cell_size, |c_end| lerp(params.cell_size, c_end, u));
        let r = w / c.max(0.001);
        let scale_strut = 3.0 * t_norm;
        (scale_strut * (r - 0.2)).clamp(-1.1 * t_norm, 1.1 * t_norm)
    };

    let t_field: Option<Vec<f64>> = if has_graded_thresh && !params.is_skeletal {
        u_field
            .as_ref()
            .map(|uf| uf.par_iter().map(|&u| sheet_t_at(u)).collect())
    } else {
        None
    };

    let t_scalar = if let (Some(qv), false) = (q, params.is_skeletal) {
        bvh::compute_abs_quantile(&scalar_val, &inside_3d, qv)
    } else {
        sheet_t_at(0.0)
    };

    let skeletal_field: Option<Vec<f64>> = if has_graded_thresh && params.is_skeletal {
        u_field
            .as_ref()
            .map(|uf| uf.par_iter().map(|&u| skel_t_at(u)).collect())
    } else {
        None
    };

    let skeletal_scalar = if params.is_skeletal {
        if let Some(qv) = q {
            bvh::compute_quantile(&scalar_val, &inside_3d, qv)
        } else {
            skel_t_at(0.0)
        }
    } else {
        0.0
    };

    let extract_cavities = |mask: &[bool]| -> (Option<MeshData>, Option<MeshData>) {
        if !params.include_cavities || params.is_skeletal || is_voronoi {
            return (None, None);
        }
        let field_a: Vec<f64> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if mask[idx] {
                    let t_val = t_field.as_ref().map_or(t_scalar, |tf| tf[idx]);
                    t_val - scalar_val[idx]
                } else {
                    1.0
                }
            })
            .collect();
        let field_b: Vec<f64> = (0..total)
            .into_par_iter()
            .map(|idx| {
                if mask[idx] {
                    let t_val = t_field.as_ref().map_or(t_scalar, |tf| tf[idx]);
                    scalar_val[idx] + t_val
                } else {
                    1.0
                }
            })
            .collect();
        let mut cav_a = extract_isosurface_cartesian(&xs, &ys, &zs, &field_a);
        manifold::ensure_positive_volume(&mut cav_a);
        let mut cav_b = extract_isosurface_cartesian(&xs, &ys, &zs, &field_b);
        manifold::ensure_positive_volume(&mut cav_b);
        (Some(cav_a), Some(cav_b))
    };

    if is_shell_void {
        progress(65, "Building shell void field...");
        let dist_in = dist_inside
            .as_ref()
            .expect("dist_inside must be computed when preserve_shell=true");
        let outside_3d: Vec<bool> = inside_3d.par_iter().map(|&b| !b).collect();
        let dist_outside = edt::edt_3d(&outside_3d, nx, ny, nz, dx, dy, dz);

        let inner_mask: Vec<bool> = dist_in
            .par_iter()
            .map(|&d| d > params.shell_thickness)
            .collect();
        let field_void: Vec<f64> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let d_in = dist_in[idx];
                if d_in <= 0.0 {
                    params.shell_thickness + dist_outside[idx]
                } else {
                    let s = scalar_val[idx];
                    let vs = if params.is_skeletal {
                        let sk = skeletal_field
                            .as_ref()
                            .map_or(skeletal_scalar, |sf| sf[idx]);
                        sk - s
                    } else {
                        let t_val = t_field.as_ref().map_or(t_scalar, |tf| tf[idx]);
                        t_val - s.abs()
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
    let voxel_h = dx.max(dy).max(dz).max(1e-6);
    let sdf_norm: Vec<f64> = {
        let dist_in = dist_inside
            .take()
            .expect("dist_inside must be computed for SDF boundary");
        let outside_3d: Vec<bool> = inside_3d.par_iter().map(|&b| !b).collect();
        let dist_outside = edt::edt_3d(&outside_3d, nx, ny, nz, dx, dy, dz);
        (0..total)
            .into_par_iter()
            .map(|idx| {
                let signed = if inside_3d[idx] {
                    -(dist_in[idx] - 0.5 * voxel_h)
                } else {
                    dist_outside[idx] - 0.5 * voxel_h
                };
                signed / voxel_h
            })
            .collect()
    };

    let mut mesh = {
        let field: Vec<f64> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let tpms_f = if params.is_skeletal {
                    let sk = skeletal_field
                        .as_ref()
                        .map_or(skeletal_scalar, |sf| sf[idx]);
                    scalar_val[idx] - sk
                } else {
                    let t_val = t_field.as_ref().map_or(t_scalar, |tf| tf[idx]);
                    scalar_val[idx].abs() - t_val
                };
                tpms_f.max(sdf_norm[idx])
            })
            .collect();
        extract_isosurface_cartesian(&xs, &ys, &zs, &field)
    };
    drop(sdf_norm);
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
            for shape in [ParametricShape::Cylinder, ParametricShape::Box] {
                let params = ParametricParams {
                    tpms_type,
                    shape,
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
                    seed: 42,
                    blend_passes: 0,
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
            grading: GradingParams::default(),
            seed: 42,
            blend_passes: 0,
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

    #[test]
    fn test_shell_to_core_wall_grading() {
        let cube = box_trimesh(6.0);
        let base_params = StlParams {
            tpms_type: TpmsType::Gyroid,
            resolution: 32,
            cell_size: 4.0,
            wall_thickness: 1.0,
            bias: 0.0,
            flat_cap: false,
            preserve_shell: false,
            shell_thickness: 1.0,
            is_skeletal: false,
            infill_percent: None,
            include_cavities: true,
            grading: GradingParams {
                mode: GradingMode::ShellToCore,
                wall_thickness_end: Some(0.3),
                bias_end: None,
                cell_size_end: None,
                transition_depth: Some(4.0),
            },
            seed: 42,
            blend_passes: 0,
        };
        let out = convert_stl(&cube.vertices, &cube.faces, &base_params, |_, _| {}).unwrap();
        let vol = manifold::signed_volume(&out.mesh);
        assert!(out.mesh.n_cells() > 12);
        assert!(vol > 0.0 && vol < 12.0 * 12.0 * 12.0);
        assert!(out.cavity_a.as_ref().unwrap().n_cells() > 0);
        assert!(out.cavity_b.as_ref().unwrap().n_cells() > 0);

        // Compatible with preserve_shell and skeletal modes
        let shell_params = StlParams {
            preserve_shell: true,
            include_cavities: false,
            ..base_params.clone()
        };
        let out_shell =
            convert_stl(&cube.vertices, &cube.faces, &shell_params, |_, _| {}).unwrap();
        assert!(manifold::signed_volume(&out_shell.mesh) > 0.0);

        let skel_params = StlParams {
            is_skeletal: true,
            include_cavities: false,
            ..base_params
        };
        let out_skel =
            convert_stl(&cube.vertices, &cube.faces, &skel_params, |_, _| {}).unwrap();
        assert!(manifold::signed_volume(&out_skel.mesh) > 0.0);
    }

    #[test]
    fn test_axial_and_radial_grading() {
        let cube = box_trimesh(6.0);
        let axial_params = StlParams {
            tpms_type: TpmsType::Diamond,
            resolution: 28,
            cell_size: 4.0,
            wall_thickness: 0.9,
            bias: -0.2,
            flat_cap: false,
            preserve_shell: false,
            shell_thickness: 1.0,
            is_skeletal: false,
            infill_percent: None,
            include_cavities: false,
            grading: GradingParams {
                mode: GradingMode::AxialZ,
                wall_thickness_end: Some(0.4),
                bias_end: Some(0.3),
                cell_size_end: Some(6.0),
                transition_depth: None,
            },
            seed: 42,
            blend_passes: 0,
        };
        let out_axial =
            convert_stl(&cube.vertices, &cube.faces, &axial_params, |_, _| {}).unwrap();
        assert!(manifold::signed_volume(&out_axial.mesh) > 0.0);

        let radial_params = StlParams {
            grading: GradingParams {
                mode: GradingMode::Radial,
                wall_thickness_end: Some(0.4),
                bias_end: None,
                cell_size_end: None,
                transition_depth: None,
            },
            ..axial_params
        };
        let out_radial =
            convert_stl(&cube.vertices, &cube.faces, &radial_params, |_, _| {}).unwrap();
        assert!(manifold::signed_volume(&out_radial.mesh) > 0.0);
    }

    #[test]
    fn test_voronoi_stl_foam_and_skeletal_blended() {
        let cube = box_trimesh(6.0);
        let foam_params = StlParams {
            tpms_type: TpmsType::Voronoi,
            resolution: 32,
            cell_size: 4.0,
            wall_thickness: 1.0,
            bias: 0.0,
            flat_cap: false,
            preserve_shell: false,
            shell_thickness: 1.0,
            is_skeletal: false,
            infill_percent: None,
            include_cavities: true,
            grading: GradingParams::default(),
            seed: 42,
            blend_passes: 1,
        };
        let out_foam = convert_stl(&cube.vertices, &cube.faces, &foam_params, |_, _| {}).unwrap();
        assert!(out_foam.mesh.n_cells() > 12);
        assert!(out_foam.cavity_a.is_none());
        assert!(out_foam.cavity_b.is_none());
        assert!(manifold::signed_volume(&out_foam.mesh) > 0.0);

        let skel_params = StlParams {
            is_skeletal: true,
            preserve_shell: true,
            blend_passes: 2,
            ..foam_params
        };
        let out_skel = convert_stl(&cube.vertices, &cube.faces, &skel_params, |_, _| {}).unwrap();
        assert!(out_skel.mesh.n_cells() > 12);
        assert!(manifold::signed_volume(&out_skel.mesh) > 0.0);
    }
}
