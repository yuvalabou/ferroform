# AGENTS.md - Development & Agent Architecture Guide

## System Overview
FerroForm is a 100% pure-Rust desktop application (`ferroform-gui`) and CLI tool (`ferroform-cli`) for generating, visualizing, and exporting radially symmetrical TPMS gyroid structures and converting arbitrary 3D STL geometries into functionally graded TPMS lattice solids.

## Workspace Architecture
- `crates/ferroform-core/`: Pure Rust computational geometry engine (`tpms.rs` TPMS kernels, `bvh.rs` Rayon-parallel containment & quantile estimation, `edt.rs` 3D Euclidean Distance Transform, `grading.rs` spatially graded lattice fields, `marching.rs` welded Marching Cubes, `manifold.rs` winding repair & binary STL export, `csg.rs` Z-plane trimming & shell void subtraction, `io.rs` binary/ASCII STL reader + OBJ/3MF writers, `stats.rs` mesh metrics & density scalar evaluation, and `LatticeInfillProvider` trait for SolidCraft).
- `crates/ferroform-cli/`: Pure-Rust `clap` Command Line Interface (`ferroform-cli`).
- `crates/ferroform-gui/`: Pure-Rust `eframe`/`egui` + `wgpu` desktop GUI (`ferroform-gui`) with interactive 3D orbit/pan/zoom viewport, GPU slice-plane clipping, Z-height & infill density colormaps, Cavity A/B translucent rendering, functional grading preview, and native file dialogs (`rfd`).
- `scripts/build_release.ps1`: Pure-Cargo Windows release build & SHA-256 packaging script.

## Technical Standards
- Language: Rust (2021 edition)
- Key Crates: `rayon`, `eframe` (`wgpu`), `egui`, `glam`, `bytemuck`, `rfd`, `clap`, `serde`, `serde_json`, `zip`

## Essential Workflows
- **Run Tests**: `cargo test --workspace`
- **Run GUI**: `cargo run --release -p ferroform-gui`
- **Run CLI**: `cargo run --release -p ferroform-cli -- -o model.stl`
- **Build Release Binaries & Zip**: `.\scripts\build_release.ps1`
- **Generate Parametric Diamond**: `cargo run --release -p ferroform-cli -- -o diamond.stl --tpms diamond`
- **Convert STL with Shell**: `cargo run --release -p ferroform-cli -- -i input.stl -o gyroid_output.stl --cell-size 5.0 --preserve-shell --shell-thickness 1.0`
- **Convert STL Skeletal (Resin Drainage)**: `cargo run --release -p ferroform-cli -- -i input.stl -o gyroid_output.stl --cell-size 5.0 --skeletal --tpms i_wp`
