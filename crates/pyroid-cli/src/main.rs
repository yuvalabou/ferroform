//! Command Line Interface (`pyroid-cli`) for PyRoid.

use std::path::PathBuf;
use std::process;

use clap::Parser;
use pyroid_core::grading::{GradingMode, GradingParams};
use pyroid_core::io::{read_stl, write_3mf, write_obj};
use pyroid_core::manifold::{signed_volume_slices, write_binary_stl, DEFAULT_STL_HEADER};
use pyroid_core::stats::{compute_mesh_stats, solve_cell_size};
use pyroid_core::tpms::TpmsType;
use pyroid_core::{
    convert_stl, generate_parametric, GeneratedLattice, ParametricParams, ParametricShape,
    StlParams,
};

#[derive(Parser, Debug)]
#[command(
    name = "pyroid-cli",
    version,
    about = "PyRoid - Pure-Rust TPMS gyroid generator & volumetric STL lattice infill tool"
)]
struct Cli {
    /// Output mesh file path (.stl, .obj, or .3mf)
    #[arg(short, long)]
    output: PathBuf,

    /// TPMS scaffold geometry: gyroid, diamond, primitive, i_wp, fischer_koch_s, neovius
    #[arg(long = "tpms", alias = "tpms-type", default_value = "gyroid")]
    tpms_type: String,

    /// Input STL file path to convert volume into a TPMS lattice
    #[arg(short = 'i', long = "input-stl")]
    input_stl: Option<PathBuf>,

    /// Grid / voxel resolution
    #[arg(long, default_value_t = 80)]
    res: usize,

    /// TPMS unit cell size (mm) for STL conversion
    #[arg(long)]
    cell_size: Option<f64>,

    /// Wall thickness (mm for STL, radial thickness for parametric cylinder)
    #[arg(long)]
    wall_thickness: Option<f64>,

    /// Target solid infill percentage (1..99); auto-computes cell size if --cell-size is omitted
    #[arg(long)]
    infill: Option<f64>,

    /// Cavity A/B volume balance offset (-1.4..1.4)
    #[arg(long, default_value_t = 0.0)]
    bias: f64,

    /// Generate skeletal strut-based lattice (single continuous void)
    #[arg(long, default_value_t = false)]
    skeletal: bool,

    /// Planar slice and cap Z extremes for 3D printing
    #[arg(long, default_value_t = false)]
    flat_cap: bool,

    /// Preserve outer boundary of input STL as a solid shell
    #[arg(long, default_value_t = false)]
    preserve_shell: bool,

    /// Outer shell thickness (mm) when --preserve-shell is enabled
    #[arg(long, default_value_t = 1.0)]
    shell_thickness: f64,

    /// Spatial grading profile: none, shell_to_core, axial_x, axial_y, axial_z, radial
    #[arg(long, default_value = "none")]
    grade_mode: String,

    /// Target wall thickness (mm) at grading End
    #[arg(long)]
    grade_wall_end: Option<f64>,

    /// Target TPMS unit cell size (mm) at grading End
    #[arg(long)]
    grade_cell_end: Option<f64>,

    /// Target cavity bias at grading End
    #[arg(long)]
    grade_bias_end: Option<f64>,

    /// Transition depth (mm) for shell_to_core grading
    #[arg(long)]
    grade_depth: Option<f64>,

    /// Parametric domain shape: cylinder or box
    #[arg(long, default_value = "cylinder")]
    shape: String,

    /// Parametric X-axis length
    #[arg(long, default_value_t = 24.0)]
    a: f64,

    /// Parametric Y-axis length
    #[arg(long, default_value_t = 24.0)]
    b: f64,

    /// Parametric Z-axis length
    #[arg(long, default_value_t = 10.0)]
    c: f64,

    /// Parametric outer cylinder radius
    #[arg(long, default_value_t = 12.0)]
    r1: f64,

    /// Parametric inner core radius
    #[arg(long, default_value_t = 0.0)]
    r2: f64,

    /// Parametric angular scaling factor
    #[arg(long, default_value_t = 8.0)]
    phi_scale: f64,

    /// Parametric cell radial/angular pore frequency
    #[arg(long, default_value_t = 2.0)]
    cell_radius: f64,

    /// Parametric cell vertical pore frequency
    #[arg(long, default_value_t = 3.0)]
    cell_height: f64,

    /// Skip generating Cavity A / Cavity B meshes (faster, smaller 3MF)
    #[arg(long, default_value_t = false)]
    no_cavities: bool,

    /// Print mesh statistics as JSON to stdout
    #[arg(long, default_value_t = false)]
    stats_json: bool,
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let tpms_type = TpmsType::from_str(&cli.tpms_type)?;
    let include_cavities = !cli.no_cavities;

    let (lattice, ref_volume): (GeneratedLattice, f64) = if let Some(stl_path) = &cli.input_stl {
        eprintln!("Loading STL: {}", stl_path.display());
        let trimesh = read_stl(stl_path)?;
        let pts_f32: Vec<[f32; 3]> = trimesh
            .vertices
            .iter()
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .collect();
        let ref_vol = signed_volume_slices(&pts_f32, &trimesh.faces).abs();

        let wall_thickness = cli.wall_thickness.unwrap_or(0.8);
        let cell_size = match (cli.cell_size, cli.infill) {
            (Some(cs), _) => cs,
            (None, Some(pct)) => solve_cell_size(pct, wall_thickness, cli.skeletal, tpms_type),
            (None, None) => 5.0,
        };
        let grade_mode = GradingMode::from_str(&cli.grade_mode)?;

        let params = StlParams {
            tpms_type,
            resolution: cli.res,
            cell_size,
            wall_thickness,
            bias: cli.bias,
            flat_cap: cli.flat_cap,
            preserve_shell: cli.preserve_shell,
            shell_thickness: cli.shell_thickness,
            is_skeletal: cli.skeletal,
            infill_percent: cli.infill,
            include_cavities,
            grading: GradingParams {
                mode: grade_mode,
                wall_thickness_end: cli.grade_wall_end,
                bias_end: cli.grade_bias_end,
                cell_size_end: cli.grade_cell_end,
                transition_depth: cli.grade_depth,
            },
        };

        let out = convert_stl(&trimesh.vertices, &trimesh.faces, &params, |pct, msg| {
            eprintln!("[{pct:3}%] {msg}");
        })?;
        (out, ref_vol)
    } else {
        let shape = ParametricShape::from_str(&cli.shape)?;
        let params = ParametricParams {
            tpms_type,
            shape,
            res: cli.res,
            a: cli.a,
            b: cli.b,
            c: cli.c,
            r1: cli.r1,
            r2: cli.r2,
            phi_scale: cli.phi_scale,
            wall_thickness: cli.wall_thickness.unwrap_or(11.5),
            cell_radius: cli.cell_radius,
            cell_height: cli.cell_height,
            bias: cli.bias,
            include_cavities,
        };
        let ref_vol = params.reference_volume();
        let out = generate_parametric(&params, |pct, msg| {
            eprintln!("[{pct:3}%] {msg}");
        })?;
        (out, ref_vol)
    };

    let ext = cli
        .output
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "stl" => write_binary_stl(&lattice.mesh, &cli.output, DEFAULT_STL_HEADER)?,
        "obj" => write_obj(&lattice.mesh, &cli.output)?,
        "3mf" => {
            let mut parts: Vec<(&str, &pyroid_core::marching::MeshData)> =
                vec![("lattice", &lattice.mesh)];
            if include_cavities {
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
            }
            write_3mf(&parts, &cli.output, "millimeter")?;
        }
        other => {
            return Err(format!(
                "Unsupported output extension '.{other}'. Expected .stl, .obj, or .3mf"
            ));
        }
    }

    eprintln!("Saved mesh to {}", cli.output.display());

    let stats = compute_mesh_stats(&lattice.mesh, Some(ref_volume));
    if cli.stats_json {
        let json = serde_json::to_string_pretty(&stats)
            .map_err(|e| format!("Failed to serialize stats: {e}"))?;
        println!("{json}");
    } else {
        eprintln!(
            "Stats: {} pts, {} faces, vol={:.2}, area={:.2}, watertight={}",
            stats.n_points, stats.n_cells, stats.volume, stats.surface_area, stats.is_watertight
        );
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}
