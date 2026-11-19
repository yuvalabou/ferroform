#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Pure-Rust egui desktop application (`pyroid-gui`) for PyRoid.

mod app;
mod viewport;

use app::PyRoidApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("PyRoid - Radially Symmetrical Gyroid Generator & STL Volume Infill")
            .with_inner_size([1280.0, 750.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "PyRoid",
        options,
        Box::new(|cc| Ok(Box::new(PyRoidApp::new(cc)))),
    )
}
