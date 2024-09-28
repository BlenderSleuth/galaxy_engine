use std::ffi::CString;

use bitflags::bitflags;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use crate::engine::GalaxyEngine;

bitflags! {
    pub struct AppFlags: u32 {
        const DEBUG = 1 << 0;
    }
}

pub struct AppInfo {
    pub name: CString,
    pub version: u32,
    pub flags: AppFlags,
}

impl AppInfo {
    pub fn new(name: &str, version: u32, flags: AppFlags) -> AppInfo {
        AppInfo {
            name: CString::new(name).unwrap_or(c"Unknown".into()),
            version,
            flags,
        }
    }
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

pub struct GalaxyApp {
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

        if let Some(engine) = self.engine.as_mut() {
            unwrap_or_exit!(engine.main_loop(), "Main loop error: \n{}\nExiting.", event_loop);
        }
    }
}
