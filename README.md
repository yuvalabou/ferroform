# PyRoid

**PyRoid** (`v2.0.0`) is a **100% pure-Rust** desktop application (`pyroid-gui`) and CLI tool (`pyroid-cli`) for generating radially symmetrical TPMS gyroid structures and converting arbitrary 3D STL geometries into functionally graded TPMS lattice solids.

Powered by a multithreaded Rust computational geometry core (`pyroid-core`), an `eframe`/`egui` + `wgpu` 3D desktop interface with 4x MSAA and GPU slice-plane clipping, and a `clap` CLI, PyRoid caters to additive manufacturing (SLS/SLA/SLM/FDM), computational design, and heat-exchanger workflows with zero Python runtime dependencies.

---

## Key Features

- **Radially Symmetrical & Cartesian TPMS Engine**: Real-time generation of cylindrical or Cartesian minimal surfaces with control over angular twist (`phi_scale`), inner/outer radii, and resolution.
- **Multi-TPMS Architectures**: Gyroid (Schoen G), Diamond (Schwarz D), Primitive (Schwarz P), I-WP (Schoen I-WP), Fischer-Koch S, and Neovius (C(P)).
- **STL Volume Infill Conversion**: Convert closed 3D STL meshes into TPMS lattices with smooth SDF boundary trimming (`edt_3d`), optional outer shell preservation (`--preserve-shell`), and planar Z-capping (`--flat-cap`).
- **Functional Grading**: Spatially vary wall thickness, unit cell size, and cavity bias across `shell_to_core` (EDT), `axial_x`, `axial_y`, `axial_z`, or `radial` profiles, with quick presets and a live grading preview bar.
- **Skeletal & Sheet Lattices**: Dual-void sheet walls (with Cavity A/B inspection) or single-void skeletal struts (`--skeletal`) for SLS powder evacuation and SLA resin drainage.
- **Interactive `egui` + `wgpu` 3D Viewport**: 4x MSAA, 3-point studio lighting, Z-height & Infill Density (%) colormaps, translucent wall + Cavity A/B rendering, GPU X/Y/Z slice-plane clipping, and drag-and-drop `.stl`/`.json` loading.
- **CAD / CAM Export**: Standard-compliant binary STL (Farsoon BuildStar & Netfabb compatible), Wavefront OBJ, and compressed multi-body 3MF (`lattice`, `cavity_a`, `cavity_b`).

---

## Building & Running

Requires the Rust toolchain (`cargo` / `rustc`, 2021 edition).

```powershell
# Clone the repository
git clone https://github.com/your-username/pyroid.git
cd pyroid

# Run unit & integration tests
cargo test --workspace

# Launch the desktop GUI
cargo run --release -p pyroid-gui
```

---

## CLI Usage (`pyroid-cli`)

### 1. Generate Parametric TPMS
```powershell
cargo run --release -p pyroid-cli -- -o my_gyroid.stl --tpms diamond --res 100
```

### 2. Convert STL Volume into a TPMS Lattice
```powershell
cargo run --release -p pyroid-cli -- -i model.stl -o gyroid_model.stl --cell-size 5.0 --wall-thickness 0.8
```

### 3. Convert STL Preserving Solid Outer Shell
```powershell
cargo run --release -p pyroid-cli -- -i model.stl -o gyroid_shelled.stl --cell-size 5.0 --preserve-shell --shell-thickness 1.5
```

### 4. Skeletal Lattice (Resin Drainage) & Functional Grading
```powershell
cargo run --release -p pyroid-cli -- -i model.stl -o graded.3mf --tpms i_wp --cell-size 5.0 --skeletal --grade-mode axial_z --grade-wall-end 1.2 --stats-json
```

---

## Standalone Windows Release Packaging

Build `pyroid-gui.exe` and `pyroid-cli.exe` and package a versioned `.zip` + `.sha256` into `dist/`:

```powershell
.\scripts\build_release.ps1
```

---

## Technical Architecture

```text
pyroid/
├── Cargo.toml                  # Pure-Rust Cargo workspace (v2.0.0)
├── Cargo.lock                  # Deterministic Rust dependency lockfile
├── crates/
│   ├── pyroid-core/            # Computational geometry engine (TPMS, BVH, EDT, MC, CSG, I/O, Stats)
│   ├── pyroid-cli/             # Pure-Rust clap CLI binary
│   └── pyroid-gui/             # Pure-Rust eframe/egui + wgpu 3D desktop application
└── scripts/
    └── build_release.ps1       # Cargo release build & SHA-256 zip packaging
```
