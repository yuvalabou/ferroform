# AGENTS.md - Development & Agent Architecture Guide

## System Overview
PyRoid is a Python package, CLI tool, and Qt graphical interface for generating, visualizing, and exporting radially symmetrical gyroid structures and converting arbitrary 3D STL geometries into gyroid lattice solids.

## Package Architecture
- `crates/pyroid-core/`: Pure Rust computational geometry engine (`tpms.rs` TPMS kernels, `bvh.rs` Rayon-parallel containment & quantile estimation, `edt.rs` 3D Euclidean Distance Transform, `marching.rs` welded Marching Cubes, `manifold.rs` winding repair, and `LatticeInfillProvider` trait for SolidCraft).
- `crates/pyroid-pyo3/`: PyO3/Maturin FFI bindings exposing `pyroid_rs` (`generate_parametric_tpms`, `convert_stl_to_tpms`) with GIL release and zero-copy NumPy buffers.
- `pyroid/tpms.py`: Python TPMS metadata registry and analytical kernel descriptors for UI/validation.
- `pyroid/core.py`: `GyroidGenerator` orchestrator forwarding parametric/STL mesh generation to `pyroid_rs`.
- `pyroid/models.py`: Strongly-typed dataclasses for parameters (`GyroidParams`, `STLConversionParams`).
- `pyroid/mesh_utils.py`: STL loading, `pyroid_rs` infill bridge, `manifold3d` shell boolean subtraction, watertight STL export, and offscreen rendering.
- `pyroid/cli.py`: Command Line Interface (`pyroid-cli`).
- `pyroid/gui.py`: Desktop PyQt5 GUI with embedded 3D PyVista renderer (`pyroid`).
- `scripts/release.py`: Automated executable release pipeline (tests, compilation, packaging, SHA-256 checksums, watch mode).
- `tests/`: Automated unit and integration tests powered by `pytest`.

## Technical Standards & Package Management
- Languages: Rust (2021 edition, `rayon`, `pyo3`, `numpy`) + Python >= 3.11
- Package Manager: `uv` (fast Python environment & package manager) + `cargo`
- Dependencies: `numpy`, `pyvista`, `trimesh`, `scipy`, `manifold3d`, `PyQt5`, `pyvistaqt`, `QDarkStyle`
- Build System: PEP 621 via `pyproject.toml` (`maturin` build-backend) with `uv.lock` and `Cargo.lock`
- Code Style: PEP 8, strict type hints, explicit dataclasses; `cargo test` for Rust core.

## Essential Workflows (with `uv`)
- **Initialize & Sync Environment**: `uv venv` && `uv sync --all-extras`
- **Rebuild Rust Extension**: `uv run maturin develop --release`
- **Run Rust Core Tests**: `cargo test --workspace`
- **Run GUI**: `uv run main.py` or `uv run pyroid`
- **Run CLI**: `uv run main.py --cli --output model.stl` or `uv run pyroid-cli -o model.stl`
- **Run Tests**: `uv run pytest -v`
- **Build Executable Release**: `uv run python scripts/release.py` or `.\scripts\build_release.bat`
- **Auto-Rebuild on Code Update (Watch Mode)**: `uv run python scripts/release.py --watch`
- **Install Git Commit Auto-Release Hook**: `uv run python scripts/release.py --install-git-hook`
- **Generate Parametric Diamond**: `uv run pyroid-cli -o diamond.stl --tpms diamond`
- **Convert STL with Diamond**: `uv run pyroid-cli -i input.stl -o diamond_output.stl --tpms diamond --cell-size 5.0`
- **Convert STL with Shell**: `uv run pyroid-cli -i input.stl -o gyroid_output.stl --cell-size 5.0 --preserve-shell --shell-thickness 1.0`
- **Convert STL Skeletal (Resin Drainage)**: `uv run pyroid-cli -i input.stl -o gyroid_output.stl --cell-size 5.0 --skeletal --tpms i_wp`
