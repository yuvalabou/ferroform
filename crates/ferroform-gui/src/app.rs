//! Main egui application state and UI for FerroForm.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

use eframe::egui;
use ferroform_core::grading::{GradingMode, GradingParams};
use ferroform_core::io::{read_stl, write_3mf, write_obj};
use ferroform_core::manifold::{signed_volume_slices, write_binary_stl, DEFAULT_STL_HEADER};
use ferroform_core::marching::MeshData;
use ferroform_core::stats::{
    compute_density_scalars, compute_mesh_stats, estimate_infill, mesh_bounds, solve_cell_size,
    solve_wall_thickness, MeshStats,
};
use ferroform_core::tpms::TpmsType;
use ferroform_core::{
    convert_stl, generate_parametric, GeneratedLattice, ParametricParams, ParametricShape,
    StlParams, Trimesh,
};

use crate::viewport::{build_mesh_vertices, coolwarm_color, viridis_color, GpuViewport};

const ALL_TPMS: [TpmsType; 7] = [
    TpmsType::Gyroid,
    TpmsType::Diamond,
    TpmsType::Primitive,
    TpmsType::IWP,
    TpmsType::FischerKochS,
    TpmsType::Neovius,
    TpmsType::Voronoi,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Stl,
    Parametric,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    HeightZ,
    Density,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GradePreset {
    None,
    StrongShellLightCore,
    LightShellStrongCore,
    HeavyBottomLightTop,
    StiffCenterSoftEdge,
    Custom,
}

impl GradePreset {
    fn label(&self) -> &'static str {
        match self {
            Self::None => "None (Uniform)",
            Self::StrongShellLightCore => "Strong Shell -> Light Core",
            Self::LightShellStrongCore => "Light Shell -> Strong Core",
            Self::HeavyBottomLightTop => "Heavy Bottom -> Light Top (Z)",
            Self::StiffCenterSoftEdge => "Stiff Center -> Soft Edge (XY)",
            Self::Custom => "Custom...",
        }
    }
}

enum WorkerMsg {
    Progress(i32, String),
    Done(Box<WorkerResult>),
    Error(String),
}

enum DialogAction {
    OpenStl,
    LoadPreset,
    SavePreset(ParametricParams),
    ExportMesh(String),
}

struct WorkerResult {
    lattice: GeneratedLattice,
    stats: MeshStats,
    stl_params: Option<StlParams>,
    input_bounds: Option<([f64; 3], [f64; 3])>,
    surface_points: Option<Vec<[f32; 3]>>,
}

struct ParametricFields {
    preset_idx: usize,
    tpms_type: TpmsType,
    shape: ParametricShape,
    res: String,
    a: String,
    b: String,
    c: String,
    r1: String,
    r2: String,
    phi_scale: String,
    wall_thickness: String,
    cell_count_xy: String,
    cell_count_z: String,
    bias: String,
    seed: String,
    blend_passes: usize,
}

impl Default for ParametricFields {
    fn default() -> Self {
        let mut s = Self {
            preset_idx: 0,
            tpms_type: TpmsType::Gyroid,
            shape: ParametricShape::Cylinder,
            res: String::new(),
            a: String::new(),
            b: String::new(),
            c: String::new(),
            r1: String::new(),
            r2: String::new(),
            phi_scale: String::new(),
            wall_thickness: String::new(),
            cell_count_xy: String::new(),
            cell_count_z: String::new(),
            bias: String::new(),
            seed: "42".to_string(),
            blend_passes: 0,
        };
        s.apply_params(&ParametricParams::default());
        s
    }
}

impl ParametricFields {
    fn apply_params(&mut self, p: &ParametricParams) {
        self.tpms_type = p.tpms_type;
        self.shape = p.shape;
        self.res = p.res.to_string();
        self.a = format!("{}", p.a);
        self.b = format!("{}", p.b);
        self.c = format!("{}", p.c);
        self.r1 = format!("{}", p.r1);
        self.r2 = format!("{}", p.r2);
        self.phi_scale = format!("{}", p.phi_scale);
        self.wall_thickness = format!("{}", p.wall_thickness);
        self.cell_count_xy = format!("{}", p.cell_count_xy);
        self.cell_count_z = format!("{}", p.cell_count_z);
        self.bias = format!("{}", p.bias);
        self.seed = p.seed.to_string();
        self.blend_passes = p.blend_passes;
    }

    fn to_params(&self) -> Result<ParametricParams, String> {
        let parse_f = |s: &str, name: &str| -> Result<f64, String> {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid number for {name}: '{s}'"))
        };
        let res = self
            .res
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid integer for Resolution: '{}'", self.res))?;
        let seed = self
            .seed
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("Invalid integer for Seed: '{}'", self.seed))?;
        Ok(ParametricParams {
            tpms_type: self.tpms_type,
            shape: self.shape,
            res,
            a: parse_f(&self.a, "X-axis Length")?,
            b: parse_f(&self.b, "Y-axis Length")?,
            c: parse_f(&self.c, "Z-axis Length")?,
            r1: parse_f(&self.r1, "Outer Radius")?,
            r2: parse_f(&self.r2, "Inner Radius")?,
            phi_scale: parse_f(&self.phi_scale, "Angular Scaling")?,
            wall_thickness: parse_f(&self.wall_thickness, "Wall Thickness")?,
            cell_count_xy: parse_f(&self.cell_count_xy, "XY Cell Count")?,
            cell_count_z: parse_f(&self.cell_count_z, "Z Cell Count")?,
            bias: parse_f(&self.bias, "Bias")?,
            include_cavities: true,
            seed,
            blend_passes: self.blend_passes,
        })
    }
}

struct StlFields {
    stl_path: String,
    tpms_type: TpmsType,
    res: String,
    infill: String,
    wall_thickness: String,
    custom_cell_override: bool,
    cell_size: String,
    bias: String,
    skeletal: bool,
    flat_cap: bool,
    preserve_shell: bool,
    shell_thickness: String,

    grade_preset: GradePreset,
    grade_mode: GradingMode,
    grade_infill_end: String,
    grade_wall_end: String,
    grade_cell_end: String,
    grade_bias_end: String,
    grade_depth: String,
    seed: String,
    blend_passes: usize,
}

impl Default for StlFields {
    fn default() -> Self {
        Self {
            stl_path: String::new(),
            tpms_type: TpmsType::Gyroid,
            res: "80".to_string(),
            infill: "30".to_string(),
            wall_thickness: "0.8".to_string(),
            custom_cell_override: false,
            cell_size: "5.0".to_string(),
            bias: "0.0".to_string(),
            skeletal: false,
            flat_cap: false,
            preserve_shell: false,
            shell_thickness: "1.0".to_string(),
            grade_preset: GradePreset::None,
            grade_mode: GradingMode::None,
            grade_infill_end: String::new(),
            grade_wall_end: String::new(),
            grade_cell_end: String::new(),
            grade_bias_end: String::new(),
            grade_depth: String::new(),
            seed: "42".to_string(),
            blend_passes: 0,
        }
    }
}

impl StlFields {
    fn start_wall(&self) -> f64 {
        self.wall_thickness.trim().parse::<f64>().unwrap_or(0.8).max(0.05)
    }

    fn start_cell(&self) -> f64 {
        let w = self.start_wall();
        if self.custom_cell_override {
            self.cell_size.trim().parse::<f64>().unwrap_or(5.0).max(0.1)
        } else {
            let pct = self.infill.trim().parse::<f64>().unwrap_or(30.0).clamp(1.0, 99.0);
            solve_cell_size(pct, w, self.skeletal, self.tpms_type)
        }
    }

    fn start_infill(&self) -> f64 {
        if self.custom_cell_override {
            estimate_infill(self.start_cell(), self.start_wall(), self.skeletal, self.tpms_type)
        } else {
            self.infill.trim().parse::<f64>().unwrap_or(30.0).clamp(1.0, 99.0)
        }
    }

    fn start_bias(&self) -> f64 {
        self.bias.trim().parse::<f64>().unwrap_or(0.0).clamp(-1.4, 1.4)
    }

    fn to_params(&self) -> Result<StlParams, String> {
        if self.stl_path.trim().is_empty() {
            return Err("Please select an input STL file first.".to_string());
        }
        let res = self
            .res
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid Voxel Resolution: '{}'", self.res))?;
        let wall_thickness = self
            .wall_thickness
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("Invalid Wall Thickness: '{}'", self.wall_thickness))?;
        let bias = if self.skeletal {
            0.0
        } else {
            self.bias
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid Bias: '{}'", self.bias))?
        };
        let shell_thickness = self
            .shell_thickness
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("Invalid Shell Thickness: '{}'", self.shell_thickness))?;

        let (cell_size, infill_percent) = if self.custom_cell_override {
            let cs = self
                .cell_size
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid Cell Size: '{}'", self.cell_size))?;
            (cs, None)
        } else {
            let pct = self
                .infill
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Invalid Target Infill %: '{}'", self.infill))?;
            let cs = solve_cell_size(pct, wall_thickness, self.skeletal, self.tpms_type);
            (cs, Some(pct))
        };

        let opt_f = |s: &str| -> Result<Option<f64>, String> {
            let t = s.trim();
            if t.is_empty() {
                Ok(None)
            } else {
                t.parse::<f64>()
                    .map(Some)
                    .map_err(|_| format!("Invalid number: '{t}'"))
            }
        };

        let seed = self
            .seed
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("Invalid integer for Seed: '{}'", self.seed))?;

        Ok(StlParams {
            tpms_type: self.tpms_type,
            resolution: res,
            cell_size,
            wall_thickness,
            bias,
            flat_cap: self.flat_cap,
            preserve_shell: self.preserve_shell,
            shell_thickness,
            is_skeletal: self.skeletal,
            infill_percent,
            include_cavities: !self.skeletal,
            grading: GradingParams {
                mode: self.grade_mode,
                wall_thickness_end: opt_f(&self.grade_wall_end)?,
                bias_end: if self.skeletal {
                    None
                } else {
                    opt_f(&self.grade_bias_end)?
                },
                cell_size_end: opt_f(&self.grade_cell_end)?,
                transition_depth: opt_f(&self.grade_depth)?,
            },
            seed,
            blend_passes: self.blend_passes,
        })
    }
}

pub struct FerroFormApp {
    viewport: GpuViewport,
    dark_mode: bool,
    active_tab: ActiveTab,

    param: ParametricFields,
    stl: StlFields,

    // Loaded preview STL before generation
    preview_stl: Option<MeshData>,

    // Generated result
    lattice: Option<GeneratedLattice>,
    stats: Option<MeshStats>,
    last_stl_params: Option<StlParams>,
    last_input_bounds: Option<([f64; 3], [f64; 3])>,
    last_surface_points: Option<Vec<[f32; 3]>>,

    // View options
    show_wall: bool,
    color_mode: ColorMode,
    show_cav_a: bool,
    show_cav_b: bool,
    slice_enabled: bool,
    slice_axis: usize, // 0=X, 1=Y, 2=Z
    slice_reverse: bool,
    slice_pct: f32,

    // Legend range if active
    scalar_legend: Option<(&'static str, f32, f32)>,

    // Background worker & async file dialog
    worker_rx: Option<Receiver<WorkerMsg>>,
    dialog_rx: Option<Receiver<(DialogAction, PathBuf)>>,
    progress_pct: i32,
    status_msg: String,
    gen_start: Option<Instant>,
    error_popup: Option<String>,
}

fn configure_fonts_and_style(ctx: &egui::Context) {
    // Load Segoe UI on Windows for crisp 1.0x HD display rendering
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(segoe_bytes) = std::fs::read(r"C:\Windows\Fonts\segoeui.ttf") {
        fonts.font_data.insert(
            "segoe_ui".to_owned(),
            std::sync::Arc::new(egui::FontData::from_owned(segoe_bytes)),
        );
        if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
            family.insert(0, "segoe_ui".to_owned());
        }
    }
    ctx.set_fonts(fonts);

    ctx.style_mut(|style| {
        use egui::{FontFamily::Proportional, FontId, TextStyle};
        style.text_styles.insert(TextStyle::Heading, FontId::new(16.0, Proportional));
        style.text_styles.insert(TextStyle::Body, FontId::new(13.5, Proportional));
        style.text_styles.insert(TextStyle::Button, FontId::new(13.5, Proportional));
        style.text_styles.insert(TextStyle::Small, FontId::new(11.5, Proportional));
        style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    });
}

impl FerroFormApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let wgpu_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be initialized with the wgpu backend");
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        configure_fonts_and_style(&cc.egui_ctx);
        Self {
            viewport: GpuViewport::new(wgpu_state),
            dark_mode: true,
            active_tab: ActiveTab::Stl,
            param: ParametricFields::default(),
            stl: StlFields::default(),
            preview_stl: None,
            lattice: None,
            stats: None,
            last_stl_params: None,
            last_input_bounds: None,
            last_surface_points: None,
            show_wall: true,
            color_mode: ColorMode::HeightZ,
            show_cav_a: false,
            show_cav_b: false,
            slice_enabled: false,
            slice_axis: 0,
            slice_reverse: false,
            slice_pct: 50.0,
            scalar_legend: None,
            worker_rx: None,
            dialog_rx: None,
            progress_pct: 0,
            status_msg: "Ready".to_string(),
            gen_start: None,
            error_popup: None,
        }
    }

    fn is_busy(&self) -> bool {
        self.worker_rx.is_some()
    }

    fn load_stl_file(&mut self, path: &Path) {
        match read_stl(path) {
            Ok(trimesh) => {
                self.stl.stl_path = path.to_string_lossy().to_string();
                let points: Vec<[f32; 3]> = trimesh
                    .vertices
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                    .collect();
                let mesh = MeshData {
                    points,
                    faces: trimesh.faces,
                };
                self.lattice = None;
                self.stats = None;
                self.preview_stl = Some(mesh);
                self.active_tab = ActiveTab::Stl;
                self.refresh_preview_stl(true);
                self.status_msg = format!(
                    "Loaded STL: {}",
                    path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default()
                );
            }
            Err(e) => {
                self.error_popup = Some(format!("Failed to load STL:\n{e}"));
            }
        }
    }

    fn load_preset_file(&mut self, path: &Path) {
        match std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<ParametricParams>(&s).map_err(|e| e.to_string()))
        {
            Ok(p) => {
                self.param.apply_params(&p);
                self.active_tab = ActiveTab::Parametric;
                self.status_msg = format!("Loaded preset: {}", path.display());
            }
            Err(e) => {
                self.error_popup = Some(format!("Failed to load preset JSON:\n{e}"));
            }
        }
    }

    fn spawn_file_dialog(&mut self, ctx: &egui::Context, action: DialogAction) {
        if self.dialog_rx.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.dialog_rx = Some(rx);
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let picked = match &action {
                DialogAction::OpenStl => rfd::FileDialog::new()
                    .add_filter("STL Files", &["stl"])
                    .pick_file(),
                DialogAction::LoadPreset => rfd::FileDialog::new()
                    .add_filter("Preset JSON", &["json"])
                    .pick_file(),
                DialogAction::SavePreset(_) => rfd::FileDialog::new()
                    .add_filter("Preset JSON", &["json"])
                    .save_file(),
                DialogAction::ExportMesh(ext) => {
                    let label = ext.to_ascii_uppercase();
                    rfd::FileDialog::new()
                        .add_filter(format!("{label} Files"), &[ext.as_str()])
                        .save_file()
                }
            };
            if let Some(path) = picked {
                let _ = tx.send((action, path));
            }
            ctx_clone.request_repaint();
        });
    }

    fn poll_dialog(&mut self) {
        let Some(rx) = self.dialog_rx.take() else {
            return;
        };
        match rx.try_recv() {
            Ok((action, mut path)) => match action {
                DialogAction::OpenStl => self.load_stl_file(&path),
                DialogAction::LoadPreset => self.load_preset_file(&path),
                DialogAction::SavePreset(params) => {
                    if path.extension().is_none() {
                        path.set_extension("json");
                    }
                    match serde_json::to_string_pretty(&params)
                        .map_err(|e| e.to_string())
                        .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
                    {
                        Ok(()) => self.status_msg = format!("Preset saved to {}", path.display()),
                        Err(e) => self.error_popup = Some(format!("Failed to save preset:\n{e}")),
                    }
                }
                DialogAction::ExportMesh(ext) => {
                    if path.extension().is_none() {
                        path.set_extension(&ext);
                    }
                    self.write_mesh_to_path(&path, &ext);
                }
            },
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                self.dialog_rx = Some(rx);
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
        }
    }

    fn save_preset_dialog(&mut self, ctx: &egui::Context) {
        match self.param.to_params() {
            Ok(p) => self.spawn_file_dialog(ctx, DialogAction::SavePreset(p)),
            Err(e) => self.error_popup = Some(e),
        }
    }

    fn refresh_preview_stl(&mut self, reset_camera: bool) {
        if self.lattice.is_some() {
            return;
        }
        let Some(ref mesh) = self.preview_stl else {
            return;
        };
        let (min_b, max_b) = mesh_bounds(&mesh.points);
        let min_f = [min_b[0] as f32, min_b[1] as f32, min_b[2] as f32];
        let max_f = [max_b[0] as f32, max_b[1] as f32, max_b[2] as f32];

        let colors: Vec<[f32; 4]> = if self.stl.grade_mode == GradingMode::None {
            self.scalar_legend = None;
            vec![[0.365, 0.678, 0.886, 1.0]; mesh.points.len()]
        } else {
            self.scalar_legend = Some(("Grading: Start (0) -> End (1)", 0.0, 1.0));
            let ex = (max_f[0] - min_f[0]).max(1e-6);
            let ey = (max_f[1] - min_f[1]).max(1e-6);
            let ez = (max_f[2] - min_f[2]).max(1e-6);
            let cx = 0.5 * (min_f[0] + max_f[0]);
            let cy = 0.5 * (min_f[1] + max_f[1]);
            let max_r = (0.5 * ex.max(ey)).max(1e-6);

            mesh.points
                .iter()
                .map(|p| {
                    let u = match self.stl.grade_mode {
                        GradingMode::AxialX => ((p[0] - min_f[0]) / ex).clamp(0.0, 1.0),
                        GradingMode::AxialY => ((p[1] - min_f[1]) / ey).clamp(0.0, 1.0),
                        GradingMode::AxialZ => ((p[2] - min_f[2]) / ez).clamp(0.0, 1.0),
                        GradingMode::Radial => {
                            ((p[0] - cx).hypot(p[1] - cy) / max_r).clamp(0.0, 1.0)
                        }
                        GradingMode::ShellToCore | GradingMode::None => 0.0,
                    };
                    coolwarm_color(u)
                })
                .collect()
        };

        let (verts, idx) = build_mesh_vertices(mesh, &colors);
        self.viewport.set_wall(&verts, &idx);
        self.viewport.set_cavity_a(&[], &[]);
        self.viewport.set_cavity_b(&[], &[]);
        self.viewport.set_bounds_lines(min_f, max_f);
        if reset_camera {
            self.viewport.camera.fit_bounds(min_f, max_f);
        }
    }

    fn upload_generated_meshes(&mut self, reset_camera: bool) {
        let Some(ref lattice) = self.lattice else {
            return;
        };
        if lattice.mesh.points.is_empty() {
            return;
        }
        let (min_b, max_b) = mesh_bounds(&lattice.mesh.points);
        let min_f = [min_b[0] as f32, min_b[1] as f32, min_b[2] as f32];
        let max_f = [max_b[0] as f32, max_b[1] as f32, max_b[2] as f32];

        let show_cavs = self.show_cav_a || self.show_cav_b;
        let wall_colors: Vec<[f32; 4]> = if show_cavs {
            self.scalar_legend = None;
            vec![[0.80, 0.80, 0.80, 1.0]; lattice.mesh.points.len()]
        } else {
            match self.color_mode {
                ColorMode::HeightZ => {
                    self.scalar_legend = None;
                    let z_span = (max_f[2] - min_f[2]).max(1e-6);
                    lattice
                        .mesh
                        .points
                        .iter()
                        .map(|p| viridis_color((p[2] - min_f[2]) / z_span))
                        .collect()
                }
                ColorMode::Density => {
                    let scalars = compute_density_scalars(
                        &lattice.mesh,
                        self.last_stl_params.as_ref(),
                        self.last_input_bounds,
                        self.last_surface_points.as_deref(),
                        self.stats.as_ref().and_then(|s| s.porosity_pct),
                    );
                    let mut s_min = f32::INFINITY;
                    let mut s_max = f32::NEG_INFINITY;
                    for &v in &scalars {
                        s_min = s_min.min(v);
                        s_max = s_max.max(v);
                    }
                    if !s_min.is_finite() {
                        s_min = 0.0;
                        s_max = 100.0;
                    }
                    let span = (s_max - s_min).max(1e-3);
                    self.scalar_legend = Some(("Infill Density (%)", s_min, s_max));
                    scalars
                        .into_iter()
                        .map(|v| coolwarm_color((v - s_min) / span))
                        .collect()
                }
            }
        };

        let (wv, wi) = build_mesh_vertices(&lattice.mesh, &wall_colors);
        self.viewport.set_wall(&wv, &wi);

        if let Some(ref ca) = lattice.cavity_a {
            let c = vec![[0.906, 0.298, 0.235, 1.0]; ca.points.len()];
            let (cv, ci) = build_mesh_vertices(ca, &c);
            self.viewport.set_cavity_a(&cv, &ci);
        } else {
            self.viewport.set_cavity_a(&[], &[]);
        }

        if let Some(ref cb) = lattice.cavity_b {
            let c = vec![[0.204, 0.596, 0.859, 1.0]; cb.points.len()];
            let (cv, ci) = build_mesh_vertices(cb, &c);
            self.viewport.set_cavity_b(&cv, &ci);
        } else {
            self.viewport.set_cavity_b(&[], &[]);
        }

        self.viewport.set_bounds_lines(min_f, max_f);
        if reset_camera {
            self.viewport.camera.fit_bounds(min_f, max_f);
        }
    }

    fn start_parametric_generation(&mut self, ctx: &egui::Context) {
        let params = match self.param.to_params() {
            Ok(p) => p,
            Err(e) => {
                self.error_popup = Some(e);
                return;
            }
        };
        if let Err(e) = params.validate() {
            self.error_popup = Some(e);
            return;
        }

        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress_pct = 0;
        self.status_msg = "Generating parametric TPMS...".to_string();
        self.gen_start = Some(Instant::now());
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let ref_vol = params.reference_volume();
            let res = generate_parametric(&params, |pct, msg| {
                let _ = tx.send(WorkerMsg::Progress(pct, msg.to_string()));
                ctx_clone.request_repaint();
            });
            match res {
                Ok(lattice) => {
                    let stats = compute_mesh_stats(&lattice.mesh, Some(ref_vol));
                    let _ = tx.send(WorkerMsg::Done(Box::new(WorkerResult {
                        lattice,
                        stats,
                        stl_params: None,
                        input_bounds: None,
                        surface_points: None,
                    })));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Error(e));
                }
            }
            ctx_clone.request_repaint();
        });
    }

    fn start_stl_conversion(&mut self, ctx: &egui::Context) {
        let params = match self.stl.to_params() {
            Ok(p) => p,
            Err(e) => {
                self.error_popup = Some(e);
                return;
            }
        };
        if let Err(e) = params.validate() {
            self.error_popup = Some(e);
            return;
        }
        let stl_path = PathBuf::from(self.stl.stl_path.trim());
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress_pct = 0;
        self.status_msg = "Loading STL & converting volume...".to_string();
        self.gen_start = Some(Instant::now());
        let ctx_clone = ctx.clone();

        std::thread::spawn(move || {
            let _ = tx.send(WorkerMsg::Progress(5, "Loading STL geometry...".to_string()));
            ctx_clone.request_repaint();

            let trimesh: Trimesh = match read_stl(&stl_path) {
                Ok(m) => m,
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Error(e));
                    ctx_clone.request_repaint();
                    return;
                }
            };

            let all_pts: Vec<[f32; 3]> = trimesh
                .vertices
                .iter()
                .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                .collect();
            let ref_vol = signed_volume_slices(&all_pts, &trimesh.faces).abs();
            let input_bounds = mesh_bounds(&all_pts);

            // Downsample surface points to <= 20k for fast density scalar preview
            let step = (all_pts.len() / 20_000).max(1);
            let surface_points: Vec<[f32; 3]> = all_pts.iter().step_by(step).copied().collect();

            let res = convert_stl(&trimesh.vertices, &trimesh.faces, &params, |pct, msg| {
                let _ = tx.send(WorkerMsg::Progress(pct, msg.to_string()));
                ctx_clone.request_repaint();
            });
            match res {
                Ok(lattice) => {
                    let stats = compute_mesh_stats(&lattice.mesh, Some(ref_vol));
                    let _ = tx.send(WorkerMsg::Done(Box::new(WorkerResult {
                        lattice,
                        stats,
                        stl_params: Some(params),
                        input_bounds: Some(input_bounds),
                        surface_points: Some(surface_points),
                    })));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Error(e));
                }
            }
            ctx_clone.request_repaint();
        });
    }

    fn export_mesh(&mut self, ctx: &egui::Context, ext: &str) {
        if self.lattice.is_none() {
            self.error_popup = Some("Please generate a mesh first.".to_string());
            return;
        }
        self.spawn_file_dialog(ctx, DialogAction::ExportMesh(ext.to_string()));
    }

    fn write_mesh_to_path(&mut self, path: &Path, ext: &str) {
        let Some(ref lattice) = self.lattice else {
            return;
        };
        let res = match ext {
            "stl" => write_binary_stl(&lattice.mesh, path, DEFAULT_STL_HEADER),
            "obj" => write_obj(&lattice.mesh, path),
            "3mf" => {
                let mut parts: Vec<(&str, &MeshData)> = vec![("lattice", &lattice.mesh)];
                if let Some(ref ca) = lattice.cavity_a {
                    if !ca.faces.is_empty() {
                        parts.push(("cavity_a", ca));
                    }
                }
                if let Some(ref cb) = lattice.cavity_b {
                    if !cb.faces.is_empty() {
                        parts.push(("cavity_b", cb));
                    }
                }
                write_3mf(&parts, path, "millimeter")
            }
            _ => Err("Unsupported export format".to_string()),
        };
        match res {
            Ok(()) => self.status_msg = format!("Saved to {}", path.display()),
            Err(e) => self.error_popup = Some(format!("Export failed:\n{e}")),
        }
    }

    fn new_project(&mut self) {
        self.lattice = None;
        self.stats = None;
        self.preview_stl = None;
        self.scalar_legend = None;
        self.param = ParametricFields::default();
        self.stl = StlFields::default();
        self.show_wall = true;
        self.show_cav_a = false;
        self.show_cav_b = false;
        self.slice_enabled = false;
        self.slice_axis = 0;
        self.slice_reverse = false;
        self.slice_pct = 50.0;
        self.viewport.clear();
        self.status_msg = "New project started".to_string();
    }

    fn active_clip_plane(&self) -> Option<([f32; 3], [f32; 3])> {
        if !self.slice_enabled {
            return None;
        }
        let mesh = self
            .lattice
            .as_ref()
            .map(|l| &l.mesh)
            .or(self.preview_stl.as_ref())?;
        if mesh.points.is_empty() {
            return None;
        }
        let (min_b, max_b) = mesh_bounds(&mesh.points);
        let sign = if self.slice_reverse { -1.0f32 } else { 1.0f32 };
        let normal = match self.slice_axis {
            0 => [sign, 0.0, 0.0],
            1 => [0.0, sign, 0.0],
            _ => [0.0, 0.0, sign],
        };
        let t = (self.slice_pct / 100.0) as f64;
        let mut origin = [
            (0.5 * (min_b[0] + max_b[0])) as f32,
            (0.5 * (min_b[1] + max_b[1])) as f32,
            (0.5 * (min_b[2] + max_b[2])) as f32,
        ];
        origin[self.slice_axis] = (min_b[self.slice_axis]
            + t * (max_b[self.slice_axis] - min_b[self.slice_axis]))
            as f32;
        Some((normal, origin))
    }

    fn apply_grade_preset(&mut self, preset: GradePreset) {
        self.stl.grade_preset = preset;
        let w = self.stl.start_wall();
        let c = self.stl.start_cell();
        let b = self.stl.start_bias();
        let skel = self.stl.skeletal;

        match preset {
            GradePreset::None => {
                self.stl.grade_mode = GradingMode::None;
                self.stl.grade_infill_end.clear();
                self.stl.grade_wall_end.clear();
                self.stl.grade_cell_end.clear();
                self.stl.grade_bias_end.clear();
                self.stl.grade_depth.clear();
            }
            GradePreset::StrongShellLightCore => {
                self.stl.grade_mode = GradingMode::ShellToCore;
                let w_end = (w * 0.45).max(0.2);
                let c_end = (c * 1.4).max(1.0);
                self.stl.grade_wall_end = format!("{w_end:.2}");
                self.stl.grade_cell_end = format!("{c_end:.2}");
                self.stl.grade_bias_end = if skel { String::new() } else { format!("{b:.2}") };
                self.stl.grade_infill_end = format!(
                    "{:.0}",
                    estimate_infill(c_end, w_end, skel, self.stl.tpms_type)
                );
            }
            GradePreset::LightShellStrongCore => {
                self.stl.grade_mode = GradingMode::ShellToCore;
                let w_end = (w * 1.8).max(0.4);
                let c_end = (c * 0.85).max(1.0);
                self.stl.grade_wall_end = format!("{w_end:.2}");
                self.stl.grade_cell_end = format!("{c_end:.2}");
                self.stl.grade_bias_end = if skel { String::new() } else { format!("{b:.2}") };
                self.stl.grade_infill_end = format!(
                    "{:.0}",
                    estimate_infill(c_end, w_end, skel, self.stl.tpms_type)
                );
            }
            GradePreset::HeavyBottomLightTop => {
                self.stl.grade_mode = GradingMode::AxialZ;
                let w_end = (w * 0.4).max(0.2);
                let c_end = (c * 1.5).max(1.0);
                self.stl.grade_wall_end = format!("{w_end:.2}");
                self.stl.grade_cell_end = format!("{c_end:.2}");
                self.stl.grade_bias_end = if skel { String::new() } else { format!("{b:.2}") };
                self.stl.grade_infill_end = format!(
                    "{:.0}",
                    estimate_infill(c_end, w_end, skel, self.stl.tpms_type)
                );
            }
            GradePreset::StiffCenterSoftEdge => {
                self.stl.grade_mode = GradingMode::Radial;
                let w_end = (w * 0.45).max(0.2);
                self.stl.grade_wall_end = format!("{w_end:.2}");
                self.stl.grade_cell_end = format!("{c:.2}");
                self.stl.grade_bias_end = if skel { String::new() } else { format!("{b:.2}") };
                self.stl.grade_infill_end = format!(
                    "{:.0}",
                    estimate_infill(c, w_end, skel, self.stl.tpms_type)
                );
            }
            GradePreset::Custom => {
                if self.stl.grade_mode == GradingMode::None {
                    self.stl.grade_mode = GradingMode::AxialZ;
                }
            }
        }
        self.refresh_preview_stl(false);
    }

    fn draw_grading_preview_bar(&self, ui: &mut egui::Ui) {
        let desired = egui::vec2(ui.available_width(), 26.0);
        let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);

        if self.stl.grade_mode == GradingMode::None {
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(44, 62, 80));
            painter.rect_stroke(
                rect,
                4.0,
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(74, 96, 122)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Uniform (no spatial grading)",
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(149, 165, 166),
            );
            return;
        }

        // Gradient background approximation via vertical strips
        let strips = 24;
        let w = rect.width() / strips as f32;
        for i in 0..strips {
            let u = i as f32 / (strips - 1) as f32;
            let r = (52.0 + (230.0 - 52.0) * u) as u8;
            let g = (152.0 + (126.0 - 152.0) * u) as u8;
            let b = (219.0 + (34.0 - 219.0) * u) as u8;
            let sub = egui::Rect::from_min_size(
                egui::pos2(rect.left() + i as f32 * w, rect.top()),
                egui::vec2(w + 0.5, rect.height()),
            );
            painter.rect_filled(sub, 0.0, egui::Color32::from_rgba_unmultiplied(r, g, b, 130));
        }

        let w0 = self.stl.start_wall();
        let w1 = self
            .stl
            .grade_wall_end
            .trim()
            .parse::<f64>()
            .unwrap_or(w0)
            .max(0.05);
        let c0 = self.stl.start_cell();
        let c1 = self
            .stl
            .grade_cell_end
            .trim()
            .parse::<f64>()
            .unwrap_or(c0)
            .max(0.5);
        let max_w = w0.max(w1).max(0.1);
        let min_c = c0.min(c1).max(0.5);

        let mut x = rect.left() + 8.0;
        let right = rect.right() - 8.0;
        let span = (right - x).max(1.0);
        while x <= right {
            let u = ((x - rect.left() - 8.0) / span) as f64;
            let wu = w0 + (w1 - w0) * u;
            let cu = c0 + (c1 - c0) * u;
            let stroke_w = (1.0 + 3.5 * (wu / max_w)).clamp(1.0, 5.0) as f32;
            painter.line_segment(
                [egui::pos2(x, rect.top() + 4.0), egui::pos2(x, rect.bottom() - 4.0)],
                egui::Stroke::new(stroke_w, egui::Color32::from_rgba_unmultiplied(236, 240, 241, 180)),
            );
            let step = (9.0 * (cu / min_c)).clamp(5.0, 22.0) as f32;
            x += step;
        }

        let (start_lbl, end_lbl) = match self.stl.grade_mode {
            GradingMode::ShellToCore => ("Outer", "Core"),
            GradingMode::AxialX => ("Left", "Right"),
            GradingMode::AxialY => ("Front", "Back"),
            GradingMode::AxialZ => ("Bottom", "Top"),
            GradingMode::Radial => ("Center", "Edge"),
            GradingMode::None => ("Start", "End"),
        };
        painter.text(
            egui::pos2(rect.left() + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            start_lbl,
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(236, 240, 241),
        );
        painter.text(
            egui::pos2(rect.right() - 6.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            end_lbl,
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(236, 240, 241),
        );
    }

    fn poll_worker(&mut self) {
        let Some(rx) = self.worker_rx.take() else {
            return;
        };
        let mut finished = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                WorkerMsg::Progress(pct, text) => {
                    self.progress_pct = pct;
                    self.status_msg = text;
                }
                WorkerMsg::Done(res) => {
                    let elapsed = self
                        .gen_start
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(0.0);
                    self.lattice = Some(res.lattice);
                    self.stats = Some(res.stats);
                    self.last_stl_params = res.stl_params;
                    self.last_input_bounds = res.input_bounds;
                    self.last_surface_points = res.surface_points;
                    self.preview_stl = None;
                    self.upload_generated_meshes(true);
                    self.status_msg = format!("Generated successfully in {elapsed:.2}s");
                    self.progress_pct = 100;
                    finished = true;
                }
                WorkerMsg::Error(err) => {
                    self.error_popup = Some(err);
                    self.status_msg = "Error during generation".to_string();
                    finished = true;
                }
            }
        }
        if finished {
            self.gen_start = None;
        } else {
            self.worker_rx = Some(rx);
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for path in dropped {
            match path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .as_deref()
            {
                Some("stl") => self.load_stl_file(&path),
                Some("json") => self.load_preset_file(&path),
                _ => {
                    self.error_popup = Some(format!(
                        "Unsupported dropped file: {}\nExpected .stl or .json preset.",
                        path.display()
                    ));
                }
            }
        }
    }
}

impl eframe::App for FerroFormApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.poll_dialog();
        self.handle_dropped_files(ctx);

        // Top Menu Bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Project").clicked() {
                        self.new_project();
                        ui.close_menu();
                    }
                    if ui.button("Open STL...").clicked() {
                        ui.close_menu();
                        self.spawn_file_dialog(ctx, DialogAction::OpenStl);
                    }
                    if ui.button("Load Preset JSON...").clicked() {
                        ui.close_menu();
                        self.spawn_file_dialog(ctx, DialogAction::LoadPreset);
                    }
                    if ui.button("Save Preset JSON...").clicked() {
                        ui.close_menu();
                        self.save_preset_dialog(ctx);
                    }
                    ui.separator();
                    if ui.button("Export as STL").clicked() {
                        ui.close_menu();
                        self.export_mesh(ctx, "stl");
                    }
                    if ui.button("Export as OBJ").clicked() {
                        ui.close_menu();
                        self.export_mesh(ctx, "obj");
                    }
                    if ui.button("Export as 3MF").clicked() {
                        ui.close_menu();
                        self.export_mesh(ctx, "3mf");
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui.button("Reset Camera").clicked() {
                        if let Some(ref l) = self.lattice {
                            let (min_b, max_b) = mesh_bounds(&l.mesh.points);
                            self.viewport.camera.fit_bounds(
                                [min_b[0] as f32, min_b[1] as f32, min_b[2] as f32],
                                [max_b[0] as f32, max_b[1] as f32, max_b[2] as f32],
                            );
                            self.viewport.dirty = true;
                        } else if let Some(ref m) = self.preview_stl {
                            let (min_b, max_b) = mesh_bounds(&m.points);
                            self.viewport.camera.fit_bounds(
                                [min_b[0] as f32, min_b[1] as f32, min_b[2] as f32],
                                [max_b[0] as f32, max_b[1] as f32, max_b[2] as f32],
                            );
                            self.viewport.dirty = true;
                        }
                        ui.close_menu();
                    }
                    let theme_lbl = if self.dark_mode {
                        "Switch to Light Theme"
                    } else {
                        "Switch to Dark Theme"
                    };
                    if ui.button(theme_lbl).clicked() {
                        self.dark_mode = !self.dark_mode;
                        ctx.set_visuals(if self.dark_mode {
                            egui::Visuals::dark()
                        } else {
                            egui::Visuals::light()
                        });
                        self.viewport.dirty = true;
                        ui.close_menu();
                    }
                });
            });
        });

        // Bottom Status Bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_msg);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_busy() {
                        let frac = (self.progress_pct as f32 / 100.0).clamp(0.0, 1.0);
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .desired_width(180.0)
                                .show_percentage(),
                        );
                    }
                });
            });
        });

        // Left Control Panel
        egui::SidePanel::left("control_panel")
            .default_width(390.0)
            .min_width(350.0)
            .max_width(480.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.active_tab,
                            ActiveTab::Stl,
                            "STL to Gyroid Volume",
                        );
                        ui.selectable_value(
                            &mut self.active_tab,
                            ActiveTab::Parametric,
                            "Parametric Gyroid",
                        );
                    });
                    ui.separator();

                    match self.active_tab {
                        ActiveTab::Stl => self.ui_stl_tab(ui, ctx),
                        ActiveTab::Parametric => self.ui_parametric_tab(ui, ctx),
                    }

                    ui.add_space(6.0);
                    self.ui_export_box(ui, ctx);
                    ui.add_space(6.0);
                    self.ui_view_box(ui);
                    if self.stats.is_some() {
                        ui.add_space(6.0);
                        self.ui_stats_box(ui);
                    }
                });
            });

        // Central 3D Viewport
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let ppp = ctx.pixels_per_point();
                let pw = (avail.x * ppp).round().max(16.0) as u32;
                let ph = (avail.y * ppp).round().max(16.0) as u32;

                let clip = self.active_clip_plane();
                self.viewport.render_if_needed(
                    pw,
                    ph,
                    self.dark_mode,
                    self.show_wall,
                    self.show_cav_a,
                    self.show_cav_b,
                    clip,
                );

                if let Some(tex_id) = self.viewport.tex_id {
                    let resp = ui.add(
                        egui::Image::new((tex_id, avail)).sense(egui::Sense::click_and_drag()),
                    );

                    if resp.dragged_by(egui::PointerButton::Primary) {
                        let d = resp.drag_delta();
                        self.viewport.camera.orbit(d.x, d.y);
                        self.viewport.dirty = true;
                    }
                    if resp.dragged_by(egui::PointerButton::Secondary)
                        || resp.dragged_by(egui::PointerButton::Middle)
                    {
                        let d = resp.drag_delta();
                        self.viewport.camera.pan(d.x, d.y);
                        self.viewport.dirty = true;
                    }
                    if resp.hovered() {
                        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                        if scroll.abs() > 0.01 {
                            self.viewport.camera.zoom(scroll);
                            self.viewport.dirty = true;
                        }
                    }
                    if resp.double_clicked() {
                        if let Some(ref l) = self.lattice {
                            let (min_b, max_b) = mesh_bounds(&l.mesh.points);
                            self.viewport.camera.fit_bounds(
                                [min_b[0] as f32, min_b[1] as f32, min_b[2] as f32],
                                [max_b[0] as f32, max_b[1] as f32, max_b[2] as f32],
                            );
                            self.viewport.dirty = true;
                        }
                    }
                }

                // Scalar legend overlay
                if let Some((title, lo, hi)) = self.scalar_legend {
                    let panel_rect = ui.max_rect();
                    let legend_rect = egui::Rect::from_min_size(
                        egui::pos2(panel_rect.right() - 210.0, panel_rect.bottom() - 58.0),
                        egui::vec2(196.0, 44.0),
                    );
                    let painter = ui.painter();
                    painter.rect_filled(
                        legend_rect,
                        6.0,
                        egui::Color32::from_rgba_unmultiplied(20, 24, 30, 200),
                    );
                    painter.text(
                        egui::pos2(legend_rect.center().x, legend_rect.top() + 10.0),
                        egui::Align2::CENTER_CENTER,
                        title,
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                    let bar_rect = egui::Rect::from_min_size(
                        egui::pos2(legend_rect.left() + 12.0, legend_rect.top() + 20.0),
                        egui::vec2(172.0, 10.0),
                    );
                    let steps = 20;
                    let sw = bar_rect.width() / steps as f32;
                    for i in 0..steps {
                        let c = coolwarm_color(i as f32 / (steps - 1) as f32);
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(bar_rect.left() + i as f32 * sw, bar_rect.top()),
                                egui::vec2(sw + 0.5, bar_rect.height()),
                            ),
                            0.0,
                            egui::Color32::from_rgb(
                                (c[0] * 255.0) as u8,
                                (c[1] * 255.0) as u8,
                                (c[2] * 255.0) as u8,
                            ),
                        );
                    }
                    painter.text(
                        egui::pos2(bar_rect.left(), bar_rect.bottom() + 7.0),
                        egui::Align2::LEFT_CENTER,
                        format!("{lo:.1}"),
                        egui::FontId::proportional(10.0),
                        egui::Color32::LIGHT_GRAY,
                    );
                    painter.text(
                        egui::pos2(bar_rect.right(), bar_rect.bottom() + 7.0),
                        egui::Align2::RIGHT_CENTER,
                        format!("{hi:.1}"),
                        egui::FontId::proportional(10.0),
                        egui::Color32::LIGHT_GRAY,
                    );
                }
            });

        // Error Modal
        if let Some(err) = self.error_popup.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(&err);
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        self.error_popup = None;
                    }
                });
        }
    }
}

impl FerroFormApp {
    fn ui_stl_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.group(|ui| {
            ui.strong("Input Mesh (STL)");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.stl.stl_path)
                        .hint_text("Select or drop STL file...")
                        .desired_width(ui.available_width() - 80.0),
                );
                if ui.button("Browse...").clicked() {
                    self.spawn_file_dialog(ctx, DialogAction::OpenStl);
                }
            });
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("Conversion Settings");
            egui::Grid::new("stl_settings_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("TPMS Scaffold:");
                    egui::ComboBox::from_id_salt("stl_tpms")
                        .selected_text(self.stl.tpms_type.display_name())
                        .show_ui(ui, |ui| {
                            for t in ALL_TPMS {
                                ui.selectable_value(&mut self.stl.tpms_type, t, t.display_name());
                            }
                        });
                    ui.end_row();
                });

            ui.add(
                egui::Label::new(
                    egui::RichText::new(self.stl.tpms_type.description())
                        .small()
                        .color(egui::Color32::from_rgb(170, 178, 189)),
                )
                .wrap(),
            );

            egui::Grid::new("stl_params_grid")
                .num_columns(2)
                .spacing([8.0, 5.0])
                .show(ui, |ui| {
                    ui.label("Voxel Resolution:");
                    ui.add(egui::TextEdit::singleline(&mut self.stl.res).desired_width(110.0));
                    ui.end_row();

                    ui.add_enabled_ui(!self.stl.custom_cell_override, |ui| {
                        ui.label("Target Infill (%):");
                    });
                    ui.add_enabled_ui(!self.stl.custom_cell_override, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.stl.infill).desired_width(110.0),
                        );
                    });
                    ui.end_row();

                    ui.label("Wall Thickness (mm):");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.stl.wall_thickness)
                            .desired_width(110.0),
                    );
                    ui.end_row();

                    ui.checkbox(&mut self.stl.custom_cell_override, "Custom Cell Size (mm):");
                    ui.add_enabled_ui(self.stl.custom_cell_override, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.stl.cell_size)
                                .desired_width(110.0),
                        );
                    });
                    ui.end_row();

                    ui.add_enabled_ui(!self.stl.skeletal, |ui| {
                        ui.label("Cavity A/B Bias (-1..1):");
                    });
                    ui.add_enabled_ui(!self.stl.skeletal, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.stl.bias).desired_width(110.0),
                        );
                    });
                    ui.end_row();

                    if self.stl.tpms_type == TpmsType::Voronoi {
                        ui.label("Poisson Seed:");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.stl.seed).desired_width(68.0),
                            );
                            if ui.button("Rnd").clicked() {
                                let next = self
                                    .stl
                                    .seed
                                    .trim()
                                    .parse::<u64>()
                                    .unwrap_or(42)
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                self.stl.seed = (next % 100_000).to_string();
                            }
                        });
                        ui.end_row();
                    }

                    ui.label("Blend Passes:");
                    ui.add(egui::Slider::new(&mut self.stl.blend_passes, 0..=4));
                    ui.end_row();
                });

            if ui
                .checkbox(
                    &mut self.stl.skeletal,
                    "Skeletal lattice (struts with single drain void)",
                )
                .changed()
                && self.stl.skeletal
            {
                self.show_cav_a = false;
                self.show_cav_b = false;
                self.show_wall = true;
                self.upload_generated_meshes(false);
            }

            ui.checkbox(
                &mut self.stl.flat_cap,
                "Planar slice & cap surfaces for 3D printing",
            );
            ui.checkbox(
                &mut self.stl.preserve_shell,
                "Preserve original STL outer shell",
            );

            ui.add_enabled_ui(self.stl.preserve_shell, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Shell Thickness (mm):");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.stl.shell_thickness)
                            .desired_width(110.0),
                    );
                });
            });
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("Functional Grading (Optional)");
            let presets = [
                GradePreset::None,
                GradePreset::StrongShellLightCore,
                GradePreset::LightShellStrongCore,
                GradePreset::HeavyBottomLightTop,
                GradePreset::StiffCenterSoftEdge,
                GradePreset::Custom,
            ];
            let mut chosen_preset = self.stl.grade_preset;
            egui::Grid::new("grade_combo_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Quick Preset:");
                    egui::ComboBox::from_id_salt("grade_preset")
                        .selected_text(self.stl.grade_preset.label())
                        .show_ui(ui, |ui| {
                            for p in presets {
                                ui.selectable_value(&mut chosen_preset, p, p.label());
                            }
                        });
                    ui.end_row();

                    let modes = [
                        (GradingMode::None, "None (Uniform)"),
                        (GradingMode::ShellToCore, "Outside Surface -> Inside Core (EDT)"),
                        (GradingMode::AxialX, "Left -> Right (X)"),
                        (GradingMode::AxialY, "Front -> Back (Y)"),
                        (GradingMode::AxialZ, "Bottom -> Top (Z)"),
                        (GradingMode::Radial, "Center -> Outer Edge (XY)"),
                    ];
                    let current_mode_lbl = modes
                        .iter()
                        .find(|(m, _)| *m == self.stl.grade_mode)
                        .map(|(_, l)| *l)
                        .unwrap_or("None");
                    ui.label("Direction:");
                    let mut new_mode = self.stl.grade_mode;
                    egui::ComboBox::from_id_salt("grade_mode")
                        .selected_text(current_mode_lbl)
                        .show_ui(ui, |ui| {
                            for (m, lbl) in modes {
                                ui.selectable_value(&mut new_mode, m, lbl);
                            }
                        });
                    if new_mode != self.stl.grade_mode {
                        self.stl.grade_mode = new_mode;
                        self.stl.grade_preset = if new_mode == GradingMode::None {
                            GradePreset::None
                        } else {
                            GradePreset::Custom
                        };
                        self.refresh_preview_stl(false);
                    }
                    ui.end_row();
                });

            if chosen_preset != self.stl.grade_preset {
                self.apply_grade_preset(chosen_preset);
            }

            let active = self.stl.grade_mode != GradingMode::None;
            let start_infill = self.stl.start_infill();
            let start_wall = self.stl.start_wall();
            let start_cell = self.stl.start_cell();
            let start_bias = self.stl.start_bias();

            ui.add_enabled_ui(active, |ui| {
                egui::Grid::new("grade_vals_grid")
                    .num_columns(3)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("End Infill (%):");
                        ui.label(egui::RichText::new(format!("{start_infill:.0}% ->")).small().weak());
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.stl.grade_infill_end)
                                    .hint_text("Same as Start")
                                    .desired_width(90.0),
                            )
                            .changed()
                        {
                            if let Ok(pct) = self.stl.grade_infill_end.trim().parse::<f64>() {
                                let c_end = self
                                    .stl
                                    .grade_cell_end
                                    .trim()
                                    .parse::<f64>()
                                    .unwrap_or(start_cell);
                                let w = solve_wall_thickness(
                                    pct,
                                    c_end,
                                    self.stl.skeletal,
                                    self.stl.tpms_type,
                                );
                                self.stl.grade_wall_end = format!("{w:.2}");
                            }
                            self.stl.grade_preset = GradePreset::Custom;
                        }
                        ui.end_row();

                        ui.label("Wall (mm):");
                        ui.label(egui::RichText::new(format!("{start_wall:.2} ->")).small().weak());
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.stl.grade_wall_end)
                                    .hint_text("Same as Start")
                                    .desired_width(90.0),
                            )
                            .changed()
                        {
                            self.stl.grade_preset = GradePreset::Custom;
                        }
                        ui.end_row();

                        ui.label("Cell Size (mm):");
                        ui.label(egui::RichText::new(format!("{start_cell:.2} ->")).small().weak());
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut self.stl.grade_cell_end)
                                    .hint_text("Same as Start")
                                    .desired_width(90.0),
                            )
                            .changed()
                        {
                            self.stl.grade_preset = GradePreset::Custom;
                        }
                        ui.end_row();

                        ui.add_enabled_ui(!self.stl.skeletal, |ui| {
                            ui.label("Bias (-1.4..1.4):");
                        });
                        ui.add_enabled_ui(!self.stl.skeletal, |ui| {
                            ui.label(egui::RichText::new(format!("{start_bias:.2} ->")).small().weak());
                        });
                        ui.add_enabled_ui(!self.stl.skeletal, |ui| {
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.stl.grade_bias_end)
                                        .hint_text("Same as Start")
                                        .desired_width(90.0),
                                )
                                .changed()
                            {
                                self.stl.grade_preset = GradePreset::Custom;
                            }
                        });
                        ui.end_row();
                    });

                if self.stl.grade_mode == GradingMode::ShellToCore {
                    ui.horizontal(|ui| {
                        ui.label("Transition Depth (mm):");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.stl.grade_depth)
                                .hint_text("Auto (to center)")
                                .desired_width(110.0),
                        );
                    });
                }
            });

            ui.add_space(4.0);
            self.draw_grading_preview_bar(ui);
        });

        ui.add_space(6.0);
        ui.add_enabled_ui(!self.is_busy(), |ui| {
            let btn = egui::Button::new(
                egui::RichText::new("Convert STL Volume to Gyroid").strong(),
            )
            .min_size(egui::vec2(ui.available_width(), 32.0));
            if ui.add(btn).clicked() {
                self.start_stl_conversion(ctx);
            }
        });
    }

    fn ui_parametric_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let presets = ParametricParams::presets();
        ui.group(|ui| {
            ui.strong("Presets");
            ui.horizontal(|ui| {
                ui.label("Preset:");
                let mut idx = self.param.preset_idx;
                egui::ComboBox::from_id_salt("param_preset")
                    .selected_text(presets[idx].0)
                    .show_ui(ui, |ui| {
                        for (i, (name, _)) in presets.iter().enumerate() {
                            ui.selectable_value(&mut idx, i, *name);
                        }
                    });
                if idx != self.param.preset_idx {
                    self.param.preset_idx = idx;
                    self.param.apply_params(&presets[idx].1);
                }
                if ui.button("Save Preset...").clicked() {
                    self.save_preset_dialog(ctx);
                }
            });
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("Scaffold Geometry");
            egui::Grid::new("param_geom_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("TPMS Type:");
                    egui::ComboBox::from_id_salt("param_tpms")
                        .selected_text(self.param.tpms_type.display_name())
                        .show_ui(ui, |ui| {
                            for t in ALL_TPMS {
                                ui.selectable_value(
                                    &mut self.param.tpms_type,
                                    t,
                                    t.display_name(),
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("Domain Shape:");
                    let shape_lbl = match self.param.shape {
                        ParametricShape::Cylinder => "Cylinder (Radial)",
                        ParametricShape::Box => "Box (Cartesian)",
                    };
                    egui::ComboBox::from_id_salt("param_shape")
                        .selected_text(shape_lbl)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.param.shape,
                                ParametricShape::Cylinder,
                                "Cylinder (Radial)",
                            );
                            ui.selectable_value(
                                &mut self.param.shape,
                                ParametricShape::Box,
                                "Box (Cartesian)",
                            );
                        });
                    ui.end_row();
                });

            ui.add(
                egui::Label::new(
                    egui::RichText::new(self.param.tpms_type.description())
                        .small()
                        .color(egui::Color32::from_rgb(170, 178, 189)),
                )
                .wrap(),
            );
        });

        ui.add_space(4.0);
        ui.group(|ui| {
            ui.strong("Parameters");
            let is_cyl = self.param.shape == ParametricShape::Cylinder;
            egui::Grid::new("param_fields_grid")
                .num_columns(2)
                .spacing([8.0, 5.0])
                .show(ui, |ui| {
                    let mut row = |label: &str, val: &mut String, enabled: bool| {
                        ui.add_enabled_ui(enabled, |ui| {
                            ui.label(label);
                        });
                        ui.add_enabled_ui(enabled, |ui| {
                            ui.add(egui::TextEdit::singleline(val).desired_width(110.0));
                        });
                        ui.end_row();
                    };
                    row("Resolution:", &mut self.param.res, true);
                    row("X-axis Length:", &mut self.param.a, true);
                    row("Y-axis Length:", &mut self.param.b, true);
                    row("Z-axis Length:", &mut self.param.c, true);
                    row("Outer Radius (r1):", &mut self.param.r1, is_cyl);
                    row("Inner Radius (r2):", &mut self.param.r2, is_cyl);
                    row("Angular Scaling:", &mut self.param.phi_scale, is_cyl);
                    row("Wall Thickness:", &mut self.param.wall_thickness, is_cyl);
                    let is_voronoi = self.param.tpms_type == TpmsType::Voronoi;
                    row(
                        if is_voronoi {
                            "Approx. Cells (X):"
                        } else {
                            "XY Cell Count:"
                        },
                        &mut self.param.cell_count_xy,
                        true,
                    );
                    row("Z Cell Count:", &mut self.param.cell_count_z, !is_voronoi);
                    row("Cavity A/B Bias (-1..1):", &mut self.param.bias, true);

                    if is_voronoi {
                        ui.label("Poisson Seed:");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.param.seed).desired_width(68.0),
                            );
                            if ui.button("Rnd").clicked() {
                                let next = self
                                    .param
                                    .seed
                                    .trim()
                                    .parse::<u64>()
                                    .unwrap_or(42)
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                self.param.seed = (next % 100_000).to_string();
                            }
                        });
                        ui.end_row();
                    }

                    ui.label("Blend Passes:");
                    ui.add(egui::Slider::new(&mut self.param.blend_passes, 0..=4));
                    ui.end_row();
                });
        });

        ui.add_space(6.0);
        ui.add_enabled_ui(!self.is_busy(), |ui| {
            let btn = egui::Button::new(egui::RichText::new("Generate Gyroid").strong())
                .min_size(egui::vec2(ui.available_width(), 32.0));
            if ui.add(btn).clicked() {
                self.start_parametric_generation(ctx);
            }
        });
    }

    fn ui_export_box(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.group(|ui| {
            ui.strong("Export Options");
            ui.horizontal(|ui| {
                if ui.button("Export STL").clicked() {
                    self.export_mesh(ctx, "stl");
                }
                if ui.button("Export OBJ").clicked() {
                    self.export_mesh(ctx, "obj");
                }
                if ui.button("Export 3MF").clicked() {
                    self.export_mesh(ctx, "3mf");
                }
            });
        });
    }

    fn ui_view_box(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.strong("View Options");
            let mut reupload = false;

            if ui.checkbox(&mut self.show_wall, "Show Solid Wall").changed() {
                self.viewport.dirty = true;
            }

            ui.horizontal(|ui| {
                ui.label("Color By:");
                let lbl = match self.color_mode {
                    ColorMode::HeightZ => "Height (Z)",
                    ColorMode::Density => "Infill Density (%)",
                };
                let prev = self.color_mode;
                egui::ComboBox::from_id_salt("color_by")
                    .selected_text(lbl)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.color_mode, ColorMode::HeightZ, "Height (Z)");
                        ui.selectable_value(
                            &mut self.color_mode,
                            ColorMode::Density,
                            "Infill Density (%)",
                        );
                    });
                if self.color_mode != prev {
                    reupload = true;
                }
            });

            let cav_enabled = !(self.active_tab == ActiveTab::Stl && self.stl.skeletal);
            ui.add_enabled_ui(cav_enabled, |ui| {
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut self.show_cav_a, "Cavity A (Red)").changed() {
                        reupload = true;
                    }
                    if ui.checkbox(&mut self.show_cav_b, "Cavity B (Blue)").changed() {
                        reupload = true;
                    }
                });
            });

            if reupload {
                self.upload_generated_meshes(false);
            }

            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.slice_enabled, "Slice Plane:").changed() {
                    self.viewport.dirty = true;
                }
                ui.add_enabled_ui(self.slice_enabled, |ui| {
                    let axes = ["X", "Y", "Z"];
                    let prev_ax = self.slice_axis;
                    egui::ComboBox::from_id_salt("slice_axis")
                        .width(42.0)
                        .selected_text(axes[self.slice_axis])
                        .show_ui(ui, |ui| {
                            for (i, ax) in axes.iter().enumerate() {
                                ui.selectable_value(&mut self.slice_axis, i, *ax);
                            }
                        });
                    if self.slice_axis != prev_ax {
                        self.viewport.dirty = true;
                    }
                    if ui.checkbox(&mut self.slice_reverse, "Reverse").changed() {
                        self.viewport.dirty = true;
                    }
                });
            });

            ui.add_enabled_ui(self.slice_enabled, |ui| {
                if ui
                    .add(egui::Slider::new(&mut self.slice_pct, 0.0..=100.0).suffix("%"))
                    .changed()
                {
                    self.viewport.dirty = true;
                }
            });
        });
    }

    fn ui_stats_box(&self, ui: &mut egui::Ui) {
        let Some(ref s) = self.stats else {
            return;
        };
        ui.group(|ui| {
            ui.strong("Mesh Statistics");
            ui.label(format!("Points: {}", s.n_points));
            ui.label(format!("Cells: {}", s.n_cells));
            ui.label(format!("Volume: {:.2} cubic units", s.volume));
            ui.label(format!("Surface Area: {:.2} square units", s.surface_area));
            let porosity_str = s
                .porosity_pct
                .map(|p| format!("{p:.1}%"))
                .unwrap_or_else(|| "-".to_string());
            ui.label(format!("Porosity: {porosity_str}"));
            ui.label(format!("Surface/Volume: {:.3} unit^-1", s.sa_to_vol_ratio));
            if s.is_watertight {
                ui.label("Watertight: Yes (0 open edges)");
            } else {
                ui.label(format!("Watertight: No ({} open edges)", s.n_open_edges));
            }
        });
    }
}
