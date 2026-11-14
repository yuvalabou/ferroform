# PyRoid: Rust Backend Migration Plan & Architecture

> **Objective:** Port PyRoid's computational geometry and volumetric infill engine to pure Rust (`pyroid-core`), expose it to Python via PyO3/Maturin (`pyroid_rs`), keep the PyQt5 + PyVista desktop frontend completely intact for daily workflows, and establish an unencumbered foundation for future integration into **SolidCraft**.

---

## 1. Architectural Strategy

### The Decoupling Principle
To ensure PyRoid remains your primary, responsive daily driver while opening the door for SolidCraft:
1. **`crates/pyroid-core` (Pure Rust):** Zero Python symbols, zero PyO3, zero CPython runtime dependencies. Pure mathematical level-sets, Rayon-parallelized containment checks, distance transforms, and marching cubes. This crate will eventually be consumed directly by SolidCraft's B-Rep engine.
2. **`crates/pyroid-pyo3` (Rust/Python FFI):** Thin bridge built with **Maturin**. Handles argument decoding, drops the Python GIL (`py.allow_threads`), coordinates progress callbacks to Qt, and yields zero-copy NumPy buffers.
3. **`pyroid/` (Python Frontend):** `gui.py`, `cli.py`, and PyVista viewport logic stay in Python. The heavy compute in `core.py` and `mesh_utils.py` is replaced with one-line calls into `pyroid_rs`.

```gfm
pyroid/
├── Cargo.toml                  # Cargo workspace manifest
├── crates/
│   ├── pyroid-core/            # Standalone geometry & TPMS engine (Pure Rust)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tpms.rs         # Multi-TPMS equations (Gyroid, Diamond, Primitive, etc.)
│   │       ├── bvh.rs          # Rayon parallel containment & ray-casting
│   │       ├── edt.rs          # 3D Euclidean Distance Transform
│   │       ├── marching.rs     # Chunked parallel Marching Cubes
│   │       └── manifold.rs     # Watertight winding & boolean operations
│   └── pyroid-pyo3/            # PyO3 module bindings
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs          # Exposes pyroid_rs C-extension
├── pyroid/                     # Untouched PyQt5 / PyVista user interface
│   ├── gui.py                  # PyQt5 GUI
│   ├── cli.py                  # CLI entry point
│   ├── core.py                 # Forwarding shim calling pyroid_rs
│   └── mesh_utils.py           # Forwarding shim calling pyroid_rs
├── pyproject.toml              # Updated to use maturin build-backend
└── uv.lock
```

---

## 2. Zero-Copy Data Handoff (Rust $\to$ PyVista)

PyVista does not need complex Python objects passed back from Rust. It only requires:
1. **Points array:** Contiguous `f32` buffer of shape `(N, 3)`.
2. **Cell connectivity array:** VTK-style `i64` or `i32` buffer of shape `(M * 4,)` where each cell is formatted as `[3, v0, v1, v2]`.

### Rust Handoff Example (`pyroid-pyo3/src/lib.rs`):
```rust
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type MeshTuple<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>);

#[pyfunction]
fn convert_stl_to_tpms<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, u32>,
    params: &Bound<'py, PyDict>,
    progress_callback: Option<PyObject>,
) -> PyResult<(bool, MeshTuple<'py>, Option<MeshTuple<'py>>, Option<MeshTuple<'py>>)> {
    // 1. Parse Python parameters & mesh buffers
    let rust_params = parse_stl_params(params)?;
    let (verts, tris) = extract_mesh_buffers(vertices, faces)?;

    // 2. Compute in pure Rust releasing GIL (briefly re-acquiring GIL between stages for progress)
    let out = py
        .allow_threads(|| pyroid_core::convert_stl(&verts, &tris, &rust_params, |pct, msg| {
            if let Some(cb) = &progress_callback {
                Python::with_gil(|py| { let _ = cb.call1(py, (pct, msg)); });
            }
        }))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // 3. Zero-copy transfer to Python NumPy arrays (primary mesh + cavity_a / cavity_b)
    Ok((
        out.is_shell_void,
        mesh_to_py(py, out.mesh)?,
        out.cavity_a.map(|m| mesh_to_py(py, m)).transpose()?,
        out.cavity_b.map(|m| mesh_to_py(py, m)).transpose()?,
    ))
}
```

### Python Consumption (`pyroid/mesh_utils.py`):
```python
from dataclasses import asdict
import numpy as np
import pyvista as pv
import pyroid_rs

def convert_stl_to_gyroid(stl_mesh, params, progress_callback=None):
    is_shell_void, mesh_buf, cav_a_buf, cav_b_buf = pyroid_rs.convert_stl_to_tpms(
        np.asarray(stl_mesh.vertices, dtype=np.float64),
        np.asarray(stl_mesh.faces, dtype=np.uint32),
        asdict(params),
        progress_callback,
    )
    # Instantaneous PyVista construction (or manifold3d boolean subtraction when is_shell_void=True)
    ...
```

---

## 2.5 Implementation Contracts & Edge Cases

1. **Dual Cavities (`cavity_a`, `cavity_b`):**
   Both parametric generation and STL sheet conversion (`is_skeletal=False`) must extract and return `(wall_mesh, cavity_a, cavity_b)` in a single pass over the evaluated scalar field so `gui.py`, `test_tpms.py`, and `test_mesh_utils.py` work seamlessly.
2. **Preserved Shell Exact CAD Vertices (`< 1e-4` tolerance):**
   When `preserve_shell=True`, `pyroid-core` computes the parallel 3D Euclidean Distance Transform (`dist_inside`) and extracts the welded internal void isosurface (`max(shell_thickness - dist_inside, void_scalar) = 0`) with positive outward signed volume. Python then subtracts `m_void` from the exact outer CAD mesh (`m_outer - m_void` via `manifold3d`), preserving original CAD boundary vertices (`< 1e-4`) and watertight internal slicing holes.
3. **Marching Cubes Vertex Welding & Boundary Closure:**
   Grid-edge vertices `(i, j, k, axis)` are deduplicated during Marching Cubes and exterior voxels are clamped to `1.0` so extracted meshes are closed, 2-manifold (`is_watertight == True`), free of degenerate triangles, and report accurate `mesh.volume`.
4. **Progress Callback GIL Contract:**
   While releasing the GIL (`py.allow_threads`) during heavy Rayon computation, `pyroid-pyo3` briefly re-acquires `Python::with_gil` between pipeline stages to emit monotonically non-decreasing `(percent, message)` updates containing the expected stage keywords (`"grid"`, `"containment"`, `"isosurface"`).
5. **Parametric Curvi-Cylindrical Mapping & Gyroid Radial Kernel:**
   `Gyroid` uses `cos(X)*sin(Y) + cos(Y)*sin(Z) + cos(Z)*sin(X)` for radial evaluation while the other 5 TPMS kernels reuse their Cartesian formula. Parametric Marching Cubes interpolates edge vertices across the curvi-cylindrical `(r * cos(phi * ky), r * sin(phi * ky), z)` grid and contours the zero-level surface `raw_g = 0`.
6. **Packaging (`pyproject.toml` & `pyroid.spec`):**
   `pyproject.toml` configures `maturin` to compile `pyroid_rs` alongside the `pyroid` Python package, and `pyroid.spec` includes `pyroid_rs` in `hiddenimports` so standalone PyInstaller releases bundle the compiled extension.

---

## 3. Migration Roadmap & Execution Checklist

### Phase 1: Build System & Workspace Setup
- [x] Create `Cargo.toml` at repository root configured for multi-crate workspace.
- [x] Initialize `crates/pyroid-core` and `crates/pyroid-pyo3`.
- [x] Switch `pyproject.toml` to `maturin`:
  ```toml
  [build-system]
  requires = ["maturin>=1.5,<2.0"]
  build-backend = "maturin"
  ```
- [x] Verify local dev setup: `uv pip install maturin && maturin develop`.

### Phase 2: `pyroid-core` Geometric Kernels
- [x] Port `pyroid/tpms.py` mathematical equations into `crates/pyroid-core/src/tpms.rs`:
  - Gyroid (Schoen G), Diamond (Schwarz D), Primitive (Schwarz P), I-WP, Fischer-Koch S, Neovius.
  - Implement radial cylindrical transformations ($r, \phi, z$).
  - Add analytical gradient scaling factors and thickness multipliers.
- [x] Verify single-point and radial evaluations via `cargo test` unit tests.

### Phase 3: Spatial Containment & Euclidean Distance Field
- [ ] Implement parallel point-in-mesh containment using `parry3d` or AABB ray-traversal with `rayon`.
- [ ] Implement multithreaded 3D Euclidean Distance Transform (Felzenszwalb-Huttenlocher algorithm) for boundary shell preservation (`shell_thickness`).
- [ ] Implement fast streaming quantile estimator for automatic `infill_percent` targeting.

### Phase 4: Isosurface Extraction & Industrial Slicing Output
- [ ] Implement parallel Marching Cubes in Rust yielding indexed triangles.
- [ ] Enforce outward-pointing normals and non-degenerate manifold winding (eliminating Farsoon BuildStar slicing errors at the vertex/face emission level).
- [ ] Integrate native CSG boolean subtraction for shell boundary retention.

### Phase 5: Python Integration & GUI Verification
- [ ] Wire PyO3 extension into `pyroid/core.py` and `pyroid/mesh_utils.py`.
- [ ] Ensure PyQt5 background thread progress bar operates smoothly with Rust callback hooks.
- [ ] Run full test suite: `uv run pytest -v`.
- [ ] Test standalone build: `uv run python scripts/release.py`.

### Phase 6: SolidCraft Hooking (Future Target)
- [ ] Expose clean Rust trait in `pyroid-core`:
  ```rust
  pub trait LatticeInfillProvider {
      fn generate_infill(&self, boundary: &Trimesh, params: &InfillParams) -> Result<Trimesh, String>;
  }
  ```
- [ ] Reference `pyroid-core` from SolidCraft via Cargo path dependency.

---

## 4. Benchmark Targets

| Task | Python (Current Baseline) | Rust Target | Target Speedup |
| :--- | :--- | :--- | :--- |
| **Cylindrical Gyroid ($120^3$)** | ~4.2 s | **~0.15 s** | **~28x** |
| **STL Containment ($150^3$, 3.3M pts)** | ~32.0 s | **~0.90 s** | **~35x** |
| **STL Preserved Shell + TPMS ($150^3$)** | ~58.0 s | **~2.20 s** | **~26x** |
| **Peak Memory Consumption** | 3.5 – 8.0 GB | **< 600 MB** | **~85% Reduction** |
