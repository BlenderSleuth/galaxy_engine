// Copyright (c) 2024-2025 Ben Sutherland.

use crate::engine::GalaxyEngine;

pub trait Game {
    fn startup(&mut self, engine: &GalaxyEngine) -> anyhow::Result<()>;
    fn update(&mut self, delta_time: f32);
    fn gui_update(&mut self, _ctx: &egui::Context) {}
}
