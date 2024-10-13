#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod engine;
pub(crate) mod utils;
mod device;
mod swapchain;
mod surface;
mod buffer;
mod maths;
mod command_buffer;
mod image;
mod gpu_alloc;
mod pipeline;
mod shader;
mod mesh;
mod material;
mod uniform_buffer;
mod particles;
mod sync;
mod descriptors;
mod static_resources;
mod debug;

use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

use app::{AppFlags, AppInfo, GalaxyApp};

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error("Event loop error: {0}.")]
    EventLoopError(#[from] EventLoopError),
}

fn main() -> Result<(), MainError> {
    // Set up logging.
    let mut log_env = env_logger::Env::default()
        .write_style_or("GX_LOG_STYLE", "always");

    log_env = if cfg!(feature = "debug_info") {
        log_env.filter_or("GX_LOG_LEVEL", "debug")
    } else {
        log_env.filter_or("GX_LOG_LEVEL", "info")
    };
    env_logger::init_from_env(log_env);

    // Set up event loop.
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    #[cfg(not(feature = "debug_info"))]
    let flags = AppFlags::empty();

    // Enable debug if in debug mode.
    #[cfg(feature = "debug_info")]
    let flags = if cfg!(debug_assertions) {
        AppFlags::DEBUG
    } else {
        AppFlags::empty()
    };

    // Set up the renderer and run the application.
    let app_info = AppInfo::new("Galaxy App", ash::vk::make_api_version(0, 0, 1, 0), flags);
    let mut app = GalaxyApp::new(app_info);
    event_loop.run_app(&mut app)?;

    Ok(())
}
