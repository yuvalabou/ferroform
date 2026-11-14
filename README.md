# PyRoid

**PyRoid** is a high-performance Rust-accelerated Python application and CLI utility designed to generate radially symmetrical gyroid structures and perform volumetric TPMS infill conversion on arbitrary 3D STL models.

Powered by a multithreaded Rust computational geometry core (`pyroid-core` exposed via PyO3/Maturin as `pyroid_rs`), a dual-mode **PyQt5** desktop interface featuring an interactive **PyVista 3D viewport**, a fully scriptable **CLI**, and a standalone **Windows executable**, PyRoid caters to additive manufacturing (SLS/SLA/FDM), computational design, and material research workflows.

---

## Key Features

- **Radially Symmetrical Gyroid Engine**: Real-time generation of polar/annular gyroid structures with fine-grained control over angular frequency (`phi_scale`), inner/outer radii, and resolution.
- **STL Volume Infill Conversion**: Turn solid 3D models into triply periodic minimal surface (TPMS) gyroid lattices.
- **Multi-TPMS Architectures**: Choose between Gyroid, Diamond (Schwarz D), Primitive (Schwarz P), I-WP, Fischer-Koch S, and Neovius minimal surface families.
- **Solid Outer Boundary Preservation**: Retain a watertight, customizable outer solid shell (`--preserve-shell`) around the gyroid lattice.
- **Skeletal & Sheet Lattice Generation**: Generate either dual-void sheet walls or single-void skeletal networks (`--skeletal`) optimized for SLS powder evacuation or SLA/DLP resin drainage.
- **Flat Surface Planar Capping**: Optional boundary capping (`--flat-cap`) for clean bed adhesion and 3D print slicing.
- **Offscreen Batch Rendering**: Automatically generate PNG renders of models directly from the CLI.
- **Embedded Interactive 3D GUI**: Real-time parameter tweaking, dark mode interface, multi-angle camera views, and built-in preset management.
- **Standalone Windows Executable**: Portable binary packaging with automated release pipeline and file-watching auto-builder.

---

## Installation

PyRoid uses [`uv`](https://github.com/astral-sh/uv) and the Rust toolchain (`cargo` / `maturin`) to compile the native `pyroid_rs` extension and manage dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/pyroid.git
cd pyroid

# Initialize environment, compile Rust backend, and install all dependencies
uv venv
uv sync --all-extras
uv run maturin develop --release
```

---

## Usage

### 1. Graphical Interface (GUI)
Launch the PyQt5 interactive application:
```bash
uv run main.py
# or
uv run pyroid
```

### 2. Command Line Interface (CLI)

#### 1. Generate Parametric Gyroid
```bash
uv run pyroid-cli --output my_gyroid.stl --preset thick_walls --res 100
```

#### 2. Convert 3D STL Model Volume into a Gyroid Structure
```bash
uv run pyroid-cli --input-stl model.stl --output gyroid_model.stl --cell-size 5.0 --wall-thickness 0.5
```

#### 3. Convert STL Preserving Solid Outer Shell
```bash
uv run pyroid-cli --input-stl model.stl --output gyroid_shelled.stl --cell-size 5.0 --wall-thickness 0.5 --preserve-shell --shell-thickness 1.5
```

#### 4. Convert STL to Skeletal Lattice (Single Continuous Void for Resin Draining)
```bash
uv run pyroid-cli --input-stl model.stl --output gyroid_skeletal.stl --cell-size 5.0 --skeletal
```

#### 5. Offscreen Thumbnail Rendering & Stats
```bash
uv run pyroid-cli --input-stl input.stl --output out.stl --render screenshot.png --print-stats
```

---

## Standalone Executable & Release Automation

PyRoid includes an automated build and release pipeline located at `scripts/release.py`.

### 1. Build Executable Release
To run tests, build the standalone executable with PyInstaller, verify binary integrity, and package into a distribution `.zip` with SHA256 checksums:
```bash
uv run python scripts/release.py
# Or on Windows:
.\scripts\build_release.bat
```
Output artifacts are placed in `dist/`:
- `dist/pyroid/pyroid.exe`
- `dist/pyroid-v1.1.0-windows-x64.zip`
- `dist/pyroid-v1.1.0-windows-x64.zip.sha256`
- `dist/release-manifest.json`

### 2. Auto-Build on Code Update (Watch Mode)
Continuously watch `pyroid/`, `main.py`, and `pyproject.toml` for code changes and automatically rebuild the executable on save:
```bash
uv run python scripts/release.py --watch
```

### 3. Auto-Build on Git Commit
Install the Git `post-commit` hook to automatically build releases when changes are committed:
```bash
uv run python scripts/release.py --install-git-hook
```

---

## Technical Architecture

```gfm
pyroid/
├── Cargo.toml                  # Cargo workspace manifest
├── Cargo.lock                  # Deterministic Rust dependency lockfile
├── crates/
│   ├── pyroid-core/            # Pure Rust TPMS, BVH ray-casting, 3D EDT, Marching Cubes
│   └── pyroid-pyo3/            # PyO3/Maturin bindings exposing pyroid_rs
├── pyroid/
│   ├── core.py                 # GyroidGenerator forwarding parametric/STL builds to pyroid_rs
│   ├── models.py               # GyroidParams & STLConversionParams dataclasses
│   ├── mesh_utils.py           # pyroid_rs infill bridge, manifold3d shell subtraction, STL export
│   ├── cli.py                  # Command Line Interface (pyroid-cli)
│   └── gui.py                  # PyQt5 dual-mode GUI & PyVistaQt interactor
├── scripts/
│   ├── release.py              # Automated executable release & watch pipeline
│   └── build_release.bat       # Windows convenience runner
├── .agents/skills/
│   └── release-executable/SKILL.md # Antigravity release agent skill
├── tests/                      # Automated pytest test suite
├── AGENTS.md                   # Developer & AI Agent contribution guide
├── pyproject.toml              # PEP 621 maturin build config & uv overrides
├── pyroid.spec                 # PyInstaller build specification
├── uv.lock                     # Deterministic uv lockfile
└── main.py                     # Entry point launcher
```

---

## Running Tests

Run the Rust core and Python integration test suites:

```bash
cargo test --workspace
uv run pytest -v
```

---

## License

Licensed under the MIT License - see [LICENSE](LICENSE) for details.
