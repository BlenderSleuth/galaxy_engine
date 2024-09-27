mod app;
mod engine;
mod utils;

use raw_window_handle::HasDisplayHandle;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use app::AppFlags;
use app::AppInfo;
use engine::GalaxyEngine;

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error("Event loop error: {0}.")]
    EventLoopError(#[from] EventLoopError),
}

struct GalaxyApp {
    app_info: AppInfo,
    window: Option<Window>,
    engine: Option<GalaxyEngine>,
}

impl GalaxyApp {
    fn new(app_info: AppInfo) -> Self {
        Self {
            app_info,
            window: None,
            engine: None,
        }
    }
}

impl ApplicationHandler for GalaxyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Galaxy Engine");

        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => window,
            Err(err) => {
                log::error!("Failed to create window: {err}. Exiting.");
                event_loop.exit();
                return;
            }
        };
        let display_handle = match window.display_handle() {
            Ok(handle) => handle,
            Err(err) => {
                log::error!("Failed to get display handle: {err}. Exiting.");
                event_loop.exit();
                return;
            }
        };

        self.engine = Some(match GalaxyEngine::new(&self.app_info, display_handle) {
            Ok(engine) => engine,
            Err(err) => {
                log::error!("Failed to create engine: {err}. Exiting.");
                event_loop.exit();
                return;
            }
        });

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
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
