# PyRoid v2.0 Roadmap

Architecture, performance, and feature status for the pure-Rust **v2.0.0** release.

---

## 1. Core Geometry & Performance (`crates/pyroid-core`)

- [X] **Parallel Marching Cubes Step 2 & Sparse Edge Welding** (`src/marching.rs`)
  - Parallelize vertex welding and triangle emission per slab with boundary-edge stitching.
  - Replace the dense `nx * ny * nz * 3 * 4`-byte `edge_verts` allocation with a sparse/chunked structure to support higher voxel resolutions (`res >= 250–500`) with lower peak memory.
- [X] **Functionally Graded Lattices** (`src/lib.rs`, `src/edt.rs`, `src/grading.rs`)
  - Support spatially varying `wall_thickness`, `cell_size`, and `bias` fields instead of uniform scalars.
  - Reuse `edt::edt_3d` distance fields for shell-to-core wall grading, plus axial/radial ramps for heat-exchanger and structural lightweighting workflows.
- [X] **Smooth SDF Boundary Trimming** (`src/lib.rs`, `src/edt.rs`)
  - Intersect the TPMS scalar field with a normalized signed distance field (`edt_3d`) so Marching Cubes trims smoothly along the STL boundary instead of voxel stair-stepping.
- [X] **Pure-Rust Mesh I/O & Analytics** (`src/io.rs`, `src/stats.rs`, `src/manifold.rs`)
  - Binary/ASCII STL reader with vertex welding, direct binary STL writer, Wavefront OBJ writer, compressed multi-body 3MF writer, and quantitative mesh stats (volume, surface area, porosity, open edges).
- [ ] **Cooperative Cancellation** (`src/lib.rs`)
  - Thread an `AtomicBool` cancellation token through `generate_parametric` and `convert_stl` so long high-resolution jobs can be aborted cleanly from the GUI.

---

## 2. Pure-Rust GUI & CLI (`crates/pyroid-gui`, `crates/pyroid-cli`)

- [X] **100% Python-Free `egui` + `wgpu` Desktop GUI** (`crates/pyroid-gui`)
  - Full migration off PyQt5/PyVista/Python/PyInstaller to `eframe`/`egui` + `wgpu` with 4x MSAA, 3-point studio lighting, Z-height & Infill Density (%) colormaps, Cavity A/B translucency, GPU X/Y/Z slice-plane clipping, `GradingPreviewBar`, native `rfd` file dialogs, and drag-and-drop.
- [X] **Pure-Rust `clap` CLI** (`crates/pyroid-cli`)
  - Full-parity CLI supporting parametric generation, STL conversion, functional grading, `.stl`/`.obj`/`.3mf` export, and `--stats-json`.
- [ ] **Batch & Config-Driven CLI** (`crates/pyroid-cli`)
  - Add `--batch <dir>` and `--config <manifest.json>` modes for unattended multi-part infill pipelines.
