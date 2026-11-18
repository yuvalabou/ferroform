# PyRoid v2.0 Roadmap

Planned architecture, performance, and feature improvements for the v2.0 release.

---

## 1. Core Geometry & Performance (`crates/pyroid-core`)

- [X] **Parallel Marching Cubes Step 2 & Sparse Edge Welding** (`src/marching.rs`)
  - Parallelize vertex welding and triangle emission per slab with boundary-edge stitching.
  - Replace the dense `nx * ny * nz * 3 * 4`-byte `edge_verts` allocation with a sparse/chunked structure to support higher voxel resolutions (`res >= 250–400`) with lower peak memory.
- [X] **Functionally Graded Lattices** (`src/lib.rs`, `src/edt.rs`, `src/grading.rs`)
  - Support spatially varying `wall_thickness`, `cell_size`, and `bias` fields instead of uniform scalars.
  - Reuse `edt::edt_3d` distance fields for shell-to-core wall grading, plus axial/radial ramps for heat-exchanger and structural lightweighting workflows.
- [ ] **Cooperative Cancellation** (`src/lib.rs`, `crates/pyroid-pyo3/src/lib.rs`)
  - Thread an `AtomicBool` cancellation token through `generate_parametric` and `convert_stl` so long high-resolution jobs can be aborted cleanly from the GUI.
- [X] **Direct Rust Binary STL Writer** (`src/manifold.rs` / export module)
  - Write binary STL directly from `MeshData` when `flat_cap=false` and `preserve_shell=false`, bypassing redundant `PyVista` → `Trimesh` → `NumPy` round-trips in `pyroid/mesh_utils.py`.

---

## 2. File Formats & CAD / CAM Interop

- [X] **3MF Export** (`pyroid/core.py`, `pyroid/cli.py`, `pyroid/gui.py`)
  - Export the primary lattice alongside `cavity_a` and `cavity_b` as named multi-body parts in a single compressed `.3mf` archive.

---

## 3. GUI & CLI Experience

- [X] **Interactive Slice Inspector & Manufacturability Metrics** (`pyroid/gui.py`)

  - Replace the fixed X-plane clip toggle with an X/Y/Z slice-plane slider.
  - Display measured porosity (%), surface-area-to-volume ratio, and watertightness status in the statistics panel.
- [ ] **Batch & Config-Driven CLI** (`pyroid/cli.py`)

  - Add `--batch <dir>` and `--config <manifest.json>` modes for unattended multi-part infill pipelines.
