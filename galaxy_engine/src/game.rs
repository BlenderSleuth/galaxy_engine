// Copyright (c) 2024 Ben Sutherland.

use crate::engine::{GalaxyEngine, StartupError};

pub trait Game {
    fn startup(&mut self, engine: &GalaxyEngine) -> Result<(), StartupError>;
    fn update(&mut self, delta_time: f32);
}
