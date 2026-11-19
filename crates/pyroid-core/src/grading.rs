//! Spatially varying (functionally graded) lattice field definitions and evaluation.

use std::f64::consts::PI;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tpms::TpmsType;

/// Supported spatial grading profiles for volumetric STL lattice conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradingMode {
    #[default]
    None,
    ShellToCore,
    AxialX,
    AxialY,
    AxialZ,
    Radial,
}

impl GradingMode {
    /// Parse a grading profile name with case and separator normalization.
    pub fn from_str(val: &str) -> Result<Self, String> {
        let cleaned = val
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .replace(' ', "_");
        match cleaned.as_str() {
            "" | "none" | "uniform" => Ok(Self::None),
            "shell_to_core" | "shell" | "edt" => Ok(Self::ShellToCore),
            "axial_x" | "x" => Ok(Self::AxialX),
            "axial_y" | "y" => Ok(Self::AxialY),
            "axial_z" | "z" => Ok(Self::AxialZ),
            "radial" | "r" => Ok(Self::Radial),
            _ => Err(format!(
                "Unknown grading mode: {val}. Supported: ['none', 'shell_to_core', 'axial_x', 'axial_y', 'axial_z', 'radial']"
            )),
        }
    }

    /// Canonical string identifier matching Python's `STLConversionParams.grade_mode`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ShellToCore => "shell_to_core",
            Self::AxialX => "axial_x",
            Self::AxialY => "axial_y",
            Self::AxialZ => "axial_z",
            Self::Radial => "radial",
        }
    }
}

/// Configuration for spatially varying `wall_thickness`, `bias`, and `cell_size`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GradingParams {
    pub mode: GradingMode,
    pub wall_thickness_end: Option<f64>,
    pub bias_end: Option<f64>,
    pub cell_size_end: Option<f64>,
    /// Transition depth in model units for `ShellToCore` (`None` or `<= 0` auto-scales to max interior distance).
    pub transition_depth: Option<f64>,
}

impl GradingParams {
    /// Returns `true` when a grading profile is selected and at least one target end value is set.
    pub fn is_active(&self) -> bool {
        self.mode != GradingMode::None
            && (self.wall_thickness_end.is_some()
                || self.bias_end.is_some()
                || self.cell_size_end.is_some())
    }

    /// Validate grading parameters.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(w) = self.wall_thickness_end {
            if w <= 0.0 {
                return Err("Graded end wall thickness must be positive".to_string());
            }
        }
        if let Some(b) = self.bias_end {
            if !(-1.4..=1.4).contains(&b) {
                return Err("Graded end bias must be between -1.4 and 1.4".to_string());
            }
        }
        if let Some(c) = self.cell_size_end {
            if c <= 0.0 {
                return Err("Graded end cell size must be positive".to_string());
            }
        }
        if let Some(d) = self.transition_depth {
            if d < 0.0 {
                return Err("Grading transition depth must be non-negative".to_string());
            }
        }
        Ok(())
    }
}

#[inline]
pub fn lerp(a: f64, b: f64, u: f64) -> f64 {
    a + (b - a) * u
}

/// Compute normalized grading factor `u in [0.0, 1.0]` for every voxel in C-order `(nx, ny, nz)`.
pub fn compute_grading_factors(
    params: &GradingParams,
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    min_b: [f64; 3],
    extent: [f64; 3],
    dist_inside: Option<&[f64]>,
) -> Vec<f64> {
    let nx = xs.len();
    let ny = ys.len();
    let nz = zs.len();
    let total = nx * ny * nz;

    match params.mode {
        GradingMode::None => vec![0.0; total],
        GradingMode::ShellToCore => {
            let dist = dist_inside.expect("dist_inside required for ShellToCore grading");
            let depth = match params.transition_depth {
                Some(d) if d > 0.0 => d,
                _ => dist
                    .par_iter()
                    .copied()
                    .reduce(|| 0.0_f64, f64::max)
                    .max(1e-6),
            };
            dist.par_iter().map(|&d| (d / depth).clamp(0.0, 1.0)).collect()
        }
        GradingMode::AxialX => {
            let inv_len = 1.0 / extent[0].max(1e-6);
            let mut u_field = vec![0.0; total];
            u_field
                .par_chunks_mut(ny * nz)
                .enumerate()
                .for_each(|(i, slab)| {
                    let u = ((xs[i] - min_b[0]) * inv_len).clamp(0.0, 1.0);
                    slab.fill(u);
                });
            u_field
        }
        GradingMode::AxialY => {
            let inv_len = 1.0 / extent[1].max(1e-6);
            let mut u_field = vec![0.0; total];
            u_field.par_chunks_mut(ny * nz).for_each(|slab| {
                for (j, &y) in ys.iter().enumerate() {
                    let u = ((y - min_b[1]) * inv_len).clamp(0.0, 1.0);
                    let row_off = j * nz;
                    slab[row_off..row_off + nz].fill(u);
                }
            });
            u_field
        }
        GradingMode::AxialZ => {
            let inv_len = 1.0 / extent[2].max(1e-6);
            let u_z: Vec<f64> = zs
                .iter()
                .map(|&z| ((z - min_b[2]) * inv_len).clamp(0.0, 1.0))
                .collect();
            let mut u_field = vec![0.0; total];
            u_field.par_chunks_mut(ny * nz).for_each(|slab| {
                for j in 0..ny {
                    slab[j * nz..(j + 1) * nz].copy_from_slice(&u_z);
                }
            });
            u_field
        }
        GradingMode::Radial => {
            let cx = min_b[0] + 0.5 * extent[0];
            let cy = min_b[1] + 0.5 * extent[1];
            let max_r = (0.5 * extent[0].max(extent[1])).max(1e-6);
            let mut u_field = vec![0.0; total];
            u_field
                .par_chunks_mut(ny * nz)
                .enumerate()
                .for_each(|(i, slab)| {
                    let dx = xs[i] - cx;
                    for (j, &y) in ys.iter().enumerate() {
                        let dy = y - cy;
                        let u = (dx.hypot(dy) / max_r).clamp(0.0, 1.0);
                        let row_off = j * nz;
                        slab[row_off..row_off + nz].fill(u);
                    }
                });
            u_field
        }
    }
}

/// Evaluate TPMS nodal level-set at `(x, y, z)` with optional spatially varying cell size.
///
/// For axial profiles, wavenumber `k(u)` is analytically integrated along the graded axis
/// to preserve phase continuity and avoid frequency distortion.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn eval_tpms_with_grading(
    tpms_type: TpmsType,
    x: f64,
    y: f64,
    z: f64,
    u: f64,
    cell_start: f64,
    cell_end: Option<f64>,
    mode: GradingMode,
    min_b: [f64; 3],
    extent: [f64; 3],
    center: [f64; 3],
) -> f64 {
    let Some(c_end) = cell_end else {
        let k = 2.0 * PI / cell_start.max(0.001);
        return tpms_type.eval(k * x, k * y, k * z);
    };

    let c_local = lerp(cell_start, c_end, u).max(0.001);
    let k0 = 2.0 * PI / cell_start.max(0.001);
    let k1 = 2.0 * PI / c_end.max(0.001);
    let k_local = 2.0 * PI / c_local;

    let (px, py, pz) = match mode {
        GradingMode::AxialX => (
            min_b[0] * k0 + extent[0] * (k0 * u + 0.5 * (k1 - k0) * u * u),
            k_local * (y - center[1]),
            k_local * (z - center[2]),
        ),
        GradingMode::AxialY => (
            k_local * (x - center[0]),
            min_b[1] * k0 + extent[1] * (k0 * u + 0.5 * (k1 - k0) * u * u),
            k_local * (z - center[2]),
        ),
        GradingMode::AxialZ => (
            k_local * (x - center[0]),
            k_local * (y - center[1]),
            min_b[2] * k0 + extent[2] * (k0 * u + 0.5 * (k1 - k0) * u * u),
        ),
        _ => (
            k_local * (x - center[0]),
            k_local * (y - center[1]),
            k_local * (z - center[2]),
        ),
    };

    tpms_type.eval(px, py, pz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grading_mode_parsing_and_validation() {
        assert_eq!(GradingMode::from_str("").unwrap(), GradingMode::None);
        assert_eq!(GradingMode::from_str("none").unwrap(), GradingMode::None);
        assert_eq!(
            GradingMode::from_str("shell-to-core").unwrap(),
            GradingMode::ShellToCore
        );
        assert_eq!(GradingMode::from_str("axial_z").unwrap(), GradingMode::AxialZ);
        assert_eq!(GradingMode::from_str("Radial").unwrap(), GradingMode::Radial);
        assert!(GradingMode::from_str("invalid").is_err());

        let valid = GradingParams {
            mode: GradingMode::ShellToCore,
            wall_thickness_end: Some(0.4),
            bias_end: Some(-0.5),
            cell_size_end: Some(8.0),
            transition_depth: Some(3.0),
        };
        assert!(valid.is_active());
        assert!(valid.validate().is_ok());

        let bad_wall = GradingParams {
            wall_thickness_end: Some(0.0),
            ..valid.clone()
        };
        assert!(bad_wall.validate().is_err());

        let bad_bias = GradingParams {
            bias_end: Some(2.0),
            ..valid.clone()
        };
        assert!(bad_bias.validate().is_err());
    }

    #[test]
    fn test_axial_and_radial_factors_bounds() {
        let xs = vec![-2.0, 0.0, 2.0];
        let ys = vec![-2.0, 0.0, 2.0];
        let zs = vec![0.0, 5.0, 10.0];
        let min_b = [-2.0, -2.0, 0.0];
        let extent = [4.0, 4.0, 10.0];

        let axial_z = GradingParams {
            mode: GradingMode::AxialZ,
            wall_thickness_end: Some(0.5),
            ..Default::default()
        };
        let u_z = compute_grading_factors(&axial_z, &xs, &ys, &zs, min_b, extent, None);
        assert!((u_z[0] - 0.0).abs() < 1e-12);
        assert!((u_z[1] - 0.5).abs() < 1e-12);
        assert!((u_z[2] - 1.0).abs() < 1e-12);

        let radial = GradingParams {
            mode: GradingMode::Radial,
            wall_thickness_end: Some(0.5),
            ..Default::default()
        };
        let u_r = compute_grading_factors(&radial, &xs, &ys, &zs, min_b, extent, None);
        // Center voxel (1, 1, 0) -> idx = 1*9 + 1*3 + 0 = 12
        assert!((u_r[12] - 0.0).abs() < 1e-12);
        // Corner (0, 0, 0) -> r = sqrt(8) >= 2.0, clamped to 1.0
        assert!((u_r[0] - 1.0).abs() < 1e-12);
    }
}
