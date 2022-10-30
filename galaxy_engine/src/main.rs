use env_logger;
use log::*;
use thiserror::Error;

use raw_window_handle::HasRawDisplayHandle;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

mod app;
use app::{AppFlags, AppInfo};
pub mod constants;
mod renderer;
mod utils;
use renderer::Renderer;

#[derive(Debug, Error)]
#[non_exhaustive]
enum MainError {
    #[error("renderer failed to initialise")]
    InitRenderer,

    #[error("window failed to initialise")]
    InitWindow,
}

fn main() -> Result<(), MainError> {
    // Define application
    let app_info = AppInfo {
        name: "Galaxy App",
        version: 1,
        flags: AppFlags::RAYTRACING
            | if cfg!(debug_assertions) {
                AppFlags::DEBUG
            } else {
                AppFlags::empty()
            },
    };

    // Set up logging
    let log_env = env_logger::Env::default()
        .filter_or("GX_LOG_LEVEL", "info")
        .write_style_or("GX_LOG_STYLE", "always");
    env_logger::init_from_env(log_env);

    // Setup event loop
    let event_loop = EventLoop::new();

    // Set up renderer
    let renderer = Renderer::new(&app_info, event_loop.raw_display_handle()).map_err(|e| {
        error!("Error while creating renderer: {}", e);
        MainError::InitRenderer
    })?;

    // Create window
    let window = match Window::new(&event_loop) {
        Ok(result) => result,
        Err(e) => {
            error!("Unable to create window: {}", e);
            return Err(MainError::InitWindow);
        }
    };

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                window_id,
                event: WindowEvent::CloseRequested,
            } if window_id == window.id() => {
                // Exit program
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                // Main loop
                renderer.main_loop();
            }
            _ => (),
        }
    });
}
