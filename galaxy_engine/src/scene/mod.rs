// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;

use crate::utils::{ConfigID, ConfigLoadError};

mod config;

struct Entity {
    name: ConfigID,
}

pub struct Scene {
    entities: Vec<Entity>,
}

impl Scene {
    pub fn new(config_filepath: &Path) -> Result<Self, ConfigLoadError> {
        let config_str = std::fs::read_to_string(config_filepath)?;
        let config = ron::from_str::<config::Scene>(&config_str)?;
        Ok(Self {
            entities: config
                .entities
                .into_iter()
                .map(|entity| Entity { name: entity.name })
                .collect(),
        })
    }
}
