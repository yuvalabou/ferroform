//! Pure Rust TPMS & volumetric lattice generation engine for PyRoid.

pub mod bvh;
pub mod edt;
pub mod tpms;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
