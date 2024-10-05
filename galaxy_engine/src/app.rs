use std::ffi::CString;

use bitflags::bitflags;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
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

struct FrameTimeAverage {
    frame_times: Vec<std::time::Duration>,
    frame_time_index: usize,
    ready: bool,
}

impl FrameTimeAverage {
    pub fn new(frame_time_count: usize) -> Self {
        Self {
            frame_times: vec![std::time::Duration::from_secs(0); frame_time_count],
            frame_time_index: 0,
            ready: false,
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn push(&mut self, frame_time: std::time::Duration) {
        self.frame_times[self.frame_time_index] = frame_time;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.len();
        self.ready |= self.frame_time_index == 0;
    }

    pub fn average(&self) -> std::time::Duration {
        let mut total = std::time::Duration::from_secs(0);
        for frame_time in &self.frame_times {
            total += *frame_time;
        }
        total / self.frame_times.len() as u32
    }
}

pub struct GalaxyApp {
    app_info: AppInfo,
    window: Option<Window>,
    engine: Option<GalaxyEngine>,
    last_frame_time: std::time::Instant,
    frame_time_average: FrameTimeAverage,
}

impl GalaxyApp {
    pub fn new(app_info: AppInfo) -> Self {
        Self {
            app_info,
            window: None,
            engine: None,
            last_frame_time: std::time::Instant::now(),
            frame_time_average: FrameTimeAverage::new(60),
        }
    }
}

impl ApplicationHandler for GalaxyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = unwrap_or_exit!(self.app_info.name.to_str(), "Title is not valid UTF-8: {}", event_loop);
        let window_attributes = Window::default_attributes().with_title(title);

        let window = unwrap_or_exit!(event_loop.create_window(window_attributes), "Failed to create window: {}\nExiting.", event_loop);
        let display_handle = unwrap_or_exit!(window.display_handle(), "Failed to get display handle: {}\nExiting.", event_loop);
        let window_handle = unwrap_or_exit!(window.window_handle(), "Failed to get window handle: {}\nExiting.", event_loop);

        let PhysicalSize {width, height} = window.inner_size();
        
        self.engine = Some(unwrap_or_exit!(GalaxyEngine::new(&self.app_info, display_handle, window_handle, width, height), "Failed to create engine: {}\nExiting.", event_loop));
        self.window = Some(window);
        self.last_frame_time = std::time::Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
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
                match event.logical_key {
                    Key::Named(key) => match key {
                        NamedKey::Escape => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }
        
        if let Some(engine) = self.engine.as_mut() {
            unwrap_or_exit!(engine.main_loop(), "Main loop error: {}\nExiting.", event_loop);
        }
        self.frame_time_average.push(self.last_frame_time.elapsed());
        self.last_frame_time = std::time::Instant::now();
    }
}
