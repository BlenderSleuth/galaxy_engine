// Copyright (c) 2024 Ben Sutherland.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use app::{AppFlags, AppInfo, GalaxyApp};
use galaxy_engine::{app, app_dir};
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error("Event loop error: {0}.")]
    EventLoopError(#[from] EventLoopError),
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

    // Set up the engine and run the application.
    let app_info = AppInfo::new_from_package_version("Gravity Game", flags, app_dir!());
    let mut app = GalaxyApp::new(app_info);
    event_loop.run_app(&mut app)?;

    Ok(())
}
