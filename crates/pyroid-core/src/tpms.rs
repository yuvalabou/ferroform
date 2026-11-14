//! Triply Periodic Minimal Surface (TPMS) mathematical kernels and properties.
//!
//! Defines supported TPMS geometries, nodal level-set equations, radial variants,
//! and calibrated thickness/surface factors matching `pyroid.tpms`.

/// Supported TPMS scaffold geometries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TpmsType {
    Gyroid,
    Diamond,
    Primitive,
    IWP,
    FischerKochS,
    Neovius,
}

impl TpmsType {
    /// Parse a TPMS geometry name or alias with case/separator normalization.
    pub fn from_str(val: &str) -> Result<Self, String> {
        let cleaned = val
            .trim()
            .to_ascii_lowercase()
            .replace('-', "_")
            .replace(' ', "_");
        if cleaned.is_empty() {
            return Ok(Self::Gyroid);
        }
        match cleaned.as_str() {
            "gyroid" => Ok(Self::Gyroid),
            "diamond" | "d" | "schwarz_d" => Ok(Self::Diamond),
            "primitive" | "p" | "schwarz_p" => Ok(Self::Primitive),
            "i_wp" | "iwp" | "schoen_i_wp" => Ok(Self::IWP),
            "fischer_koch_s" | "fks" | "fischer_koch" | "fischerkoch" => Ok(Self::FischerKochS),
            "neovius" | "neo" | "c_p" => Ok(Self::Neovius),
            _ => Err(format!(
                "Unknown TPMS type: {val}. Supported: ['gyroid', 'diamond', 'primitive', 'i_wp', 'fischer_koch_s', 'neovius']"
            )),
        }
    }

    /// Canonical identifier string matching Python's `TPMSType.value`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gyroid => "gyroid",
            Self::Diamond => "diamond",
            Self::Primitive => "primitive",
            Self::IWP => "i_wp",
            Self::FischerKochS => "fischer_koch_s",
            Self::Neovius => "neovius",
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Gyroid => "Gyroid (Schoen G)",
            Self::Diamond => "Diamond (Schwarz D)",
            Self::Primitive => "Primitive (Schwarz P)",
            Self::IWP => "I-WP (Schoen I-WP)",
            Self::FischerKochS => "Fischer-Koch S",
            Self::Neovius => "Neovius (C(P))",
        }
    }

    /// Empirical calibration multiplier relating physical wall thickness to level-set threshold.
    pub fn thickness_multiplier(&self) -> f64 {
        match self {
            Self::Gyroid => 4.80,
            Self::Diamond => 3.32,
            Self::Primitive => 4.12,
            Self::IWP => 13.50,
            Self::FischerKochS => 5.52,
            Self::Neovius => 8.03,
        }
    }

    /// Specific surface factor used for sheet infill percentage estimation.
    pub fn surface_factor(&self) -> f64 {
        match self {
            Self::Gyroid => 318.3,
            Self::Diamond => 384.0,
            Self::Primitive => 235.0,
            Self::IWP => 358.0,
            Self::FischerKochS => 396.0,
            Self::Neovius => 512.0,
        }
    }

    /// Engineering description of the TPMS geometry.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Gyroid => "Optimal balance of self-supporting printability, isotropic modulus, and drainability.",
            Self::Diamond => "Highest specific stiffness and energy absorption; stretching-dominated sheet deformation.",
            Self::Primitive => "Simple cubic topology with open axial channels; high axial stiffness, easy flow through axes.",
            Self::IWP => "BCC interlocking package with very high crushing plateau and multi-directional strength.",
            Self::FischerKochS => "Chiral labyrinth with twisted channels; high tortuosity and compact surface area.",
            Self::Neovius => "14-face polyhedra with ultra-high surface area; best for catalytic carriers and micro-exchangers.",
        }
    }

    /// Evaluate the 3D Cartesian TPMS nodal level-set equation at `(x, y, z)` in radians.
    #[inline]
    pub fn eval(&self, x: f64, y: f64, z: f64) -> f64 {
        match self {
            Self::Gyroid => x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos(),
            Self::Diamond => {
                x.cos() * y.cos() * z.cos() - x.sin() * y.sin() * z.sin()
            }
            Self::Primitive => x.cos() + y.cos() + z.cos(),
            Self::IWP => {
                let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
                2.0 * (cx * cy + cy * cz + cz * cx)
                    - ((2.0 * x).cos() + (2.0 * y).cos() + (2.0 * z).cos())
            }
            Self::FischerKochS => {
                (2.0 * x).cos() * y.sin() * z.cos()
                    + x.cos() * (2.0 * y).cos() * z.sin()
                    + x.sin() * y.cos() * (2.0 * z).cos()
            }
            Self::Neovius => {
                let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
                3.0 * (cx + cy + cz) + 4.0 * (cx * cy * cz)
            }
        }
    }

    /// Evaluate the radial TPMS scalar field in cylindrical transformed space.
    #[inline]
    pub fn eval_radial(&self, x: f64, y: f64, z: f64) -> f64 {
        match self {
            Self::Gyroid => x.cos() * y.sin() + y.cos() * z.sin() + z.cos() * x.sin(),
            _ => self.eval(x, y, z),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_from_str_aliases() {
        assert_eq!(TpmsType::from_str("").unwrap(), TpmsType::Gyroid);
        assert_eq!(TpmsType::from_str("gyroid").unwrap(), TpmsType::Gyroid);
        assert_eq!(TpmsType::from_str("Diamond").unwrap(), TpmsType::Diamond);
        assert_eq!(TpmsType::from_str("d").unwrap(), TpmsType::Diamond);
        assert_eq!(TpmsType::from_str("schwarz-d").unwrap(), TpmsType::Diamond);
        assert_eq!(TpmsType::from_str("P").unwrap(), TpmsType::Primitive);
        assert_eq!(TpmsType::from_str("iwp").unwrap(), TpmsType::IWP);
        assert_eq!(TpmsType::from_str("fischer-koch").unwrap(), TpmsType::FischerKochS);
        assert_eq!(TpmsType::from_str("c_p").unwrap(), TpmsType::Neovius);
        assert!(TpmsType::from_str("unknown_tpms").is_err());
    }

    #[test]
    fn test_eval_known_values_and_finiteness() {
        let all = [
            TpmsType::Gyroid,
            TpmsType::Diamond,
            TpmsType::Primitive,
            TpmsType::IWP,
            TpmsType::FischerKochS,
            TpmsType::Neovius,
        ];

        assert!((TpmsType::Gyroid.eval(0.0, 0.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((TpmsType::Gyroid.eval(FRAC_PI_2, 0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((TpmsType::Gyroid.eval_radial(0.0, FRAC_PI_2, 0.0) - 1.0).abs() < 1e-12);
        assert!((TpmsType::Diamond.eval(0.0, 0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((TpmsType::Primitive.eval(0.0, 0.0, 0.0) - 3.0).abs() < 1e-12);
        assert!((TpmsType::IWP.eval(0.0, 0.0, 0.0) - 3.0).abs() < 1e-12);
        assert!((TpmsType::FischerKochS.eval(0.0, 0.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((TpmsType::Neovius.eval(0.0, 0.0, 0.0) - 13.0).abs() < 1e-12);

        for tpms in all {
            assert!(tpms.thickness_multiplier() > 0.0);
            assert!(tpms.surface_factor() > 0.0);
            assert!(!tpms.display_name().is_empty());
            assert!(!tpms.description().is_empty());
            for i in 0..5 {
                let c = (i as f64) * PI / 4.0;
                let v = tpms.eval(c, c * 0.5, c * 1.5);
                let vr = tpms.eval_radial(c, c * 0.5, c * 1.5);
                assert!(v.is_finite() && vr.is_finite());
            }
        }
    }
}
