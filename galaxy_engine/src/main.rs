mod app;
mod engine;
mod utils;
mod device;
mod swapchain;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
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

// Unwrap macro to log an error and exit the event loop on error.
macro_rules! unwrap_or_exit {
        ($result:expr, $message:literal, $event_loop:ident) => {
            match $result {
                Ok(value) => value,
                Err(err) => {
                    log::error!($message, err);
                    $event_loop.exit();
                    return;
                }
            }
        };
    }

struct GalaxyApp {
    app_info: AppInfo,
    window: Option<Window>,
    engine: Option<GalaxyEngine>,
}

impl GalaxyApp {
    pub fn new(app_info: AppInfo) -> Self {
        Self {
            app_info,
            window: None,
            engine: None,
        }
    }
}

impl ApplicationHandler for GalaxyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = unwrap_or_exit!(self.app_info.name.to_str(), "Title is not valid UTF-8: {}", event_loop);
        let window_attributes = Window::default_attributes().with_title(title);
        
        let window = unwrap_or_exit!(event_loop.create_window(window_attributes), "Failed to create window: \n{}\nExiting.", event_loop);
        let display_handle = unwrap_or_exit!(window.display_handle(), "Failed to get display handle: \n{}\nExiting.", event_loop);
        let window_handle = unwrap_or_exit!(window.window_handle(), "Failed to get window handle: \n{}\nExiting.", event_loop);
        
        self.engine = Some(unwrap_or_exit!(GalaxyEngine::new(&self.app_info, display_handle, window_handle, window.inner_size()), "Failed to create engine: \n{}\nExiting.", event_loop));
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
