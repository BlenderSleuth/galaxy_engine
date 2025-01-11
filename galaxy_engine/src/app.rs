// Copyright (c) 2024-2025 Ben Sutherland.

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;

pub use ash::vk::make_api_version;
use bitflags::bitflags;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::engine::GalaxyEngine;
use crate::game::Game;
use crate::gui::GuiIntegration;
use crate::utils;

bitflags! {
    pub struct AppFlags: u32 {
        #[cfg(feature = "debug_info")]
        const DEBUG = 1 << 0;
    }
}

#[macro_export]
macro_rules! game_dir {
    () => {
        //if cfg!(feature = "packaged") {
        //    std::path::PathBuf::new()
        //} else {
        std::path::PathBuf::from(env!("CARGO_PKG_NAME"))
        //}
    };
}

pub struct AppInfo {
    pub name: CString,
    pub version: u32,
    pub flags: AppFlags,
    pub game_dir: PathBuf,
}

impl AppInfo {
    pub fn new(name: &str, version: u32, flags: AppFlags, game_dir: PathBuf) -> AppInfo {
        AppInfo {
            name: CString::new(name).unwrap_or(c"Unknown".into()),
            version,
            flags,
            game_dir,
        }
    }
    pub fn new_from_package_version(name: &str, flags: AppFlags, game_dir: PathBuf) -> AppInfo {
        Self::new(name, utils::pkg_version(), flags, game_dir)
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

// Valid post-resumed event.
struct ActiveApp {
    // Importantly dropped in this order.
    egui_renderer: GuiIntegration,
    engine: GalaxyEngine,
    window: Arc<Window>,
}

pub struct GalaxyApp {
    app_info: AppInfo,
    active: Option<ActiveApp>,
    // Temporary storage for the game init data until the engine is created.
    game_temp: Option<Box<dyn Game>>,
    last_frame_time: std::time::Instant,
}

impl GalaxyApp {
    pub fn new(app_info: AppInfo, game: Box<dyn Game>) -> Self {
        Self {
            app_info,
            active: None,
            game_temp: Some(game),
            last_frame_time: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for GalaxyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = unwrap_or_exit!(self.app_info.name.to_str(), "Title is not valid UTF-8: {}", event_loop);
        let viewport_builder = egui::ViewportBuilder::default().with_title(title);

        let egui_ctx = egui::Context::default();

        let window = Arc::new(unwrap_or_exit!(
            egui_winit::create_window(&egui_ctx, event_loop, &viewport_builder,),
            "Failed to create window: {}\nExiting.",
            event_loop
        ));

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

        let engine = unwrap_or_exit!(
            GalaxyEngine::new(
                &self.app_info,
                display_handle,
                window_handle,
                width,
                height,
                self.game_temp.take().unwrap()
            ),
            "Failed to create engine: {}.\nExiting.",
            event_loop
        );

        let egui_renderer = unwrap_or_exit!(
            GuiIntegration::new(egui_ctx, Arc::clone(&window), event_loop, &engine),
            "Failed to create GUI renderer: {}.\nExiting.",
            event_loop
        );

        let active = ActiveApp {
            window,
            engine,
            egui_renderer,
        };

        self.active = Some(active);

        self.last_frame_time = std::time::Instant::now();

        unwrap_or_exit!(
            self.active.as_mut().unwrap().engine.startup(),
            "Failed to start engine: {}.\nExiting.",
            event_loop
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if event_loop.exiting() {
            return;
        }

        let Some(active) = self.active.as_mut() else {
            return;
        };

        debug_assert_eq!(_window_id, active.window.id());

        // Process egui events. Returns true if the event should be passed along to the game.
        if !active.egui_renderer.on_window_event(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                active.engine.notify_window_resize(size.width, size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                active.engine.notify_keyboard_input(&event);

                if let Key::Named(key) = event.logical_key {
                    match key {
                        NamedKey::Escape => {
                            event_loop.exit();
                        }
                        NamedKey::F11 => {
                            if event.state.is_pressed() {
                                // TODO: Set up fullscreen options for the user, and prefer borderless on macOS.
                                if active.window.fullscreen().is_none() {
                                    let monitor = active.window.current_monitor().unwrap();
                                    if let Some(video_mode) = monitor.video_modes().max_by(|a, b| {
                                        a.size()
                                            .cmp(&b.size())
                                            .then(a.refresh_rate_millihertz().cmp(&b.refresh_rate_millihertz()))
                                    }) {
                                        log::info!("Fullscreen video mode: {:?}", video_mode);
                                        active.window.set_fullscreen(Some(Fullscreen::Exclusive(video_mode)));
                                    } else {
                                        log::warn!("No fullscreen video modes found.");
                                    }
                                } else {
                                    active.window.set_fullscreen(None);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                active.engine.notify_mouse_button(state, button);
            }
            WindowEvent::RedrawRequested => {
                unwrap_or_exit!(
                    active.engine.main_loop(&mut active.egui_renderer),
                    "Main loop error: {}.\nExiting.",
                    event_loop
                );
            }
            _ => {}
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if event_loop.exiting() {
            return;
        }

        let Some(active) = self.active.as_mut() else {
            return;
        };

        match event {
            DeviceEvent::MouseMotion { delta } => {
                active.egui_renderer.on_mouse_motion(delta);
                active.engine.notify_mouse_motion(delta.0 as f32, delta.1 as f32);
            }
            DeviceEvent::MouseWheel { delta } => {
                active.engine.notify_mouse_wheel(delta);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }

        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}
