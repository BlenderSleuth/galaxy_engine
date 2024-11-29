// Copyright (c) 2024 Ben Sutherland.

use std::ffi::CString;
use std::path::{Path, PathBuf};

pub use ash::vk::make_api_version;
use bitflags::bitflags;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::engine::GalaxyEngine;
use crate::game::Game;
use crate::utils;

bitflags! {
    pub struct AppFlags: u32 {
        #[cfg(feature = "debug_info")]
        const DEBUG = 1 << 0;
    }
}

#[macro_export]
macro_rules! app_dir {
    () => {
        std::path::Path::new(env!("CARGO_PKG_NAME"))
    };
}

pub struct AppInfo {
    pub name: CString,
    pub version: u32,
    pub flags: AppFlags,
    pub dir: PathBuf,
}

impl AppInfo {
    pub fn new(name: &str, version: u32, flags: AppFlags, dir: &Path) -> AppInfo {
        AppInfo {
            name: CString::new(name).unwrap_or(c"Unknown".into()),
            version,
            flags,
            dir: dir.join(GalaxyEngine::CONTENT_DIR),
        }
    }
    pub fn new_from_package_version(name: &str, flags: AppFlags, dir: &Path) -> AppInfo {
        Self::new(name, utils::pkg_version(), flags, dir)
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
    // Temporary storage for the game init data until the engine is created.
    game_temp: Option<Box<dyn Game>>,
    last_frame_time: std::time::Instant,
}

impl GalaxyApp {
    pub fn new(app_info: AppInfo, game: Box<dyn Game>) -> Self {
        Self {
            app_info,
            window: None,
            engine: None,
            game_temp: Some(game),
            last_frame_time: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for GalaxyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = unwrap_or_exit!(self.app_info.name.to_str(), "Title is not valid UTF-8: {}", event_loop);
        let window_attributes = Window::default_attributes().with_title(title);

        let window = unwrap_or_exit!(
            event_loop.create_window(window_attributes),
            "Failed to create window: {}\nExiting.",
            event_loop
        );

        if cfg!(target_os = "macos") {
            window.set_cursor_grab(CursorGrabMode::Locked).unwrap();
        } else if cfg!(target_os = "windows") {
            window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        }
        window.set_cursor_visible(false);

        let display_handle = unwrap_or_exit!(
            window.display_handle(),
            "Failed to get display handle: {}\nExiting.",
            event_loop
        );
        let window_handle = unwrap_or_exit!(
            window.window_handle(),
            "Failed to get window handle: {}\nExiting.",
            event_loop
        );

        let PhysicalSize { width, height } = window.inner_size();

        self.engine = Some(unwrap_or_exit!(
            GalaxyEngine::new(
                &self.app_info,
                display_handle,
                window_handle,
                width,
                height,
                self.game_temp.take().unwrap()
            ),
            "Failed to create engine: {}\nExiting.",
            event_loop
        ));
        self.window = Some(window);
        self.last_frame_time = std::time::Instant::now();

        if let Some(engine) = self.engine.as_mut() {
            unwrap_or_exit!(engine.startup(), "Failed to start engine: {}\nExiting.", event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if event_loop.exiting() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.notify_window_resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.notify_keyboard_input(&event);
                }

                match event.logical_key {
                    Key::Named(key) => match key {
                        NamedKey::Escape => {
                            event_loop.exit();
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.notify_mouse_button(state, button);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(engine) = self.engine.as_mut() {
                    unwrap_or_exit!(engine.main_loop(), "Main loop error: {}.\nExiting.", event_loop);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if event_loop.exiting() {
            return;
        }

        match event {
            DeviceEvent::MouseMotion { delta } => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.notify_mouse_motion(delta.0 as f32, delta.1 as f32);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }

        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
