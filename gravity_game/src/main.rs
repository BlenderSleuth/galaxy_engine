// Copyright (c) 2024-2025 Ben Sutherland.

#![cfg_attr(not(feature = "debug_info"), windows_subsystem = "windows")]

use app::{AppFlags, AppInfo, GalaxyApp};
use galaxy_engine::engine::GalaxyEngine;
use galaxy_engine::game::Game;
use galaxy_engine::level::{ComponentConfig, LoadResult, LoadingLevel};
use galaxy_engine::resource_paths::ResourcePath;
use galaxy_engine::vulkan::command_buffer::TransientPrimaryCommandPool;
use galaxy_engine::{app, game_dir, register_components};
use serde::{Deserialize, Serialize};
use shipyard::EntityId;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error("Event loop error: {0}.")]
    EventLoopError(#[from] EventLoopError),
}

#[derive(Serialize, Deserialize, Debug)]
struct GravitySourceConfig {
    strength: f32,
}

impl ComponentConfig for GravitySourceConfig {
    fn load(
        &mut self,
        _entity_id: EntityId,
        _level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        Ok(())
    }
}

register_components!(ComponentConfigEnum, GravitySource: GravitySourceConfig);

struct GravityGame {
    name: String,
    age: u32,
}

impl GravityGame {
    fn new() -> Self {
        GravityGame {
            name: "Arthur".to_string(),
            age: 42,
        }
    }
}

impl Game for GravityGame {
    fn startup(&mut self, engine: &GalaxyEngine) -> anyhow::Result<()> {
        log::info!("Gravity Game started.");

        // Load level.
        let level_path = ResourcePath::new("/game/default", None).unwrap();
        engine.load_level::<ComponentConfigEnum>(level_path)?;

        Ok(())
    }

    fn update(&mut self, _delta_time: f32) {}

    fn gui_update(&mut self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(200, 50, 50, 180);
        ctx.set_visuals(visuals);
        egui::SidePanel::new(egui::panel::Side::Right, "Panel").show(ctx, |ui| {
            ui.heading("My egui Application");
            ui.horizontal(|ui| {
                let name_label = ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name).labelled_by(name_label.id);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Increment").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{}', age {}", &self.name, self.age));
        });
    }
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

    // Set up the game and run the application.
    let app_info = AppInfo::new_from_package_version("Gravity Game", flags, game_dir!());
    let game = Box::new(GravityGame::new());
    let mut app = GalaxyApp::new(app_info, game);
    event_loop.run_app(&mut app)?;

    Ok(())
}
