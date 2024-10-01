mod app;
mod engine;
mod utils;
mod device;
mod swapchain;
mod surface;
mod buffer;
mod maths;

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
    let log_env = env_logger::Env::default()
        .filter_or("GX_LOG_LEVEL", "info")
        .write_style_or("GX_LOG_STYLE", "always");
    env_logger::init_from_env(log_env);

    // Set up event loop.
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // Set up the renderer and run the application.
    let flags = if cfg!(debug_assertions) {
        AppFlags::DEBUG
    } else {
        AppFlags::empty()
    };

    let app_info = AppInfo::new("Galaxy App", ash::vk::make_api_version(0, 0, 1, 0), flags);
    let mut app = GalaxyApp::new(app_info);
    event_loop.run_app(&mut app)?;

    Ok(())
}
