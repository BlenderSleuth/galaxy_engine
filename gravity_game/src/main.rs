// Copyright (c) 2024 Ben Sutherland.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::Path;

use app::{AppFlags, AppInfo, GalaxyApp};
use galaxy_engine::engine::GalaxyEngine;
use galaxy_engine::game::Game;
use galaxy_engine::{app, app_dir};
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error("Event loop error: {0}.")]
    EventLoopError(#[from] EventLoopError),
}

struct GravityGame {}

impl GravityGame {
    fn new() -> Self {
        GravityGame {}
    }
}

impl Game for GravityGame {
    fn startup(&mut self, engine: &GalaxyEngine) -> anyhow::Result<()> {
        log::info!("Gravity Game started.");

        // Load level.
        engine.load_scene(Path::new("default.level.ron"))?;

        Ok(())
    }

    fn update(&mut self, _delta_time: f32) {}
}

fn main() -> Result<(), MainError> {
    // Set up logging.
    let mut log_env = env_logger::Env::default().write_style_or("GX_LOG_STYLE", "always");

    log_env = if cfg!(feature = "debug_info") {
        log_env.filter_or("GX_LOG_LEVEL", "debug")
    } else {
        log_env.filter_or("GX_LOG_LEVEL", "warn")
    };
    env_logger::init_from_env(log_env);

    // Set up event loop.
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // Enable debug if in debug mode.
    #[cfg(feature = "debug_info")]
    let flags = AppFlags::DEBUG;
    #[cfg(not(feature = "debug_info"))]
    let flags = AppFlags::empty();

    // Set up the game and run the application.
    let app_info = AppInfo::new_from_package_version("Gravity Game", flags, app_dir!());
    let game = Box::new(GravityGame::new());
    let mut app = GalaxyApp::new(app_info, game);
    event_loop.run_app(&mut app)?;

    Ok(())
}
