#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Pure-Rust egui desktop application (`ferroform-gui`) for FerroForm.

mod app;
mod viewport;

use app::FerroFormApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("FerroForm - Radially Symmetrical Gyroid Generator & STL Volume Infill")
            .with_inner_size([1280.0, 750.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "FerroForm",
        options,
        Box::new(|cc| Ok(Box::new(FerroFormApp::new(cc)))),
    )
}
