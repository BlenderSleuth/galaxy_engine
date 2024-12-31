// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::HashMap;

use self_cell::self_cell;

use crate::engine::GalaxyEngine;
use crate::resource_paths::{resource_type, ResourcePath, SubresourcePath};

#[derive(serde::Deserialize, Debug, Copy, Clone)]
pub enum ResourceConstant {
    Int(i32),
    RGB(u8, u8, u8),
    Float(f32),
    Float2(f32, f32),
    Float3(f32, f32, f32),
    Float4(f32, f32, f32, f32),
}

#[derive(serde::Deserialize, Clone, Copy, Debug)]
pub enum ResourceBindingConfig<'a> {
    Texture(&'a str),
    Constant(ResourceConstant),
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename = "Material")]
pub struct MaterialConfig<'a> {
    pub pipeline: &'a str,
    pub params: HashMap<&'a str, ResourceBindingConfig<'a>>,
}

#[derive(thiserror::Error, Debug)]
pub enum MaterialConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse material config (resource: {:?}, map: {:?}, single: {:?})", .0.0, .0.1, .0.2)]
    Parse(Box<(SubresourcePath, ron::de::SpannedError, ron::de::SpannedError)>),
    #[error("Subresource not found: {0}")]
    Resource(String),
}

// Config self-referring structs.
self_cell!(
    struct MaterialConfigYoke {
        owner: String,
        #[covariant]
        dependent: MaterialConfig,
    }
);

pub type MaterialConfigMap<'a> = HashMap<&'a str, MaterialConfig<'a>>;

self_cell!(
    struct MaterialConfigMapYoke {
        owner: String,
        #[covariant]
        dependent: MaterialConfigMap,
    }
);

pub struct MaterialConfigsCache {
    config_maps: HashMap<ResourcePath, MaterialConfigMapYoke>,
    config_singles: HashMap<ResourcePath, MaterialConfigYoke>,
}

impl MaterialConfigsCache {
    pub fn new() -> Self {
        Self {
            config_maps: HashMap::new(),
            config_singles: HashMap::new(),
        }
    }

    pub fn get_or_load_material_config(
        &mut self,
        engine: &GalaxyEngine,
        subresource: &SubresourcePath,
    ) -> Result<&MaterialConfig, MaterialConfigError> {
        let resource = subresource.resource();
        let mut in_map_cache = self.config_maps.contains_key(resource);
        let mut in_single_cache = self.config_singles.contains_key(resource);
        if !in_map_cache && !in_single_cache {
            // Load config string.
            let full_path = resource.full_path::<resource_type::Material>(engine);
            let config_str = std::fs::read_to_string(full_path)?;

            match MaterialConfigMapYoke::try_new_or_recover(config_str, |config_str| ron::from_str(config_str)) {
                Ok(yoke) => {
                    self.config_maps.insert(resource.clone(), yoke);
                    in_map_cache = true;
                }
                Err((config_str, err_map)) => {
                    match MaterialConfigYoke::try_new(config_str, |config_str| ron::from_str(config_str)) {
                        Ok(yoke) => {
                            self.config_singles.insert(resource.clone(), yoke);
                            in_single_cache = true;
                        }
                        Err(err_single) => {
                            return Err(MaterialConfigError::Parse(Box::new((
                                subresource.clone(),
                                err_map,
                                err_single,
                            ))));
                        }
                    }
                }
            }
        }
        // Check the config is only loaded into one cache.
        debug_assert!(in_map_cache ^ in_single_cache);

        if in_single_cache {
            Ok(self.config_singles[resource].borrow_dependent())
        } else {
            self.config_maps[resource]
                .borrow_dependent()
                .get(subresource.subresource())
                .ok_or(MaterialConfigError::Resource(format!(
                    "Subresource not found {subresource:?}"
                )))
        }
    }
}

/*

#[derive(thiserror::Error, Debug)]
pub enum MaterialConfigError {
    #[error("Unknown pipeline: {0}")]
    UnknownPipeline(ConfigID),
}

pub trait MaterialConfig: std::fmt::Debug {
    fn pipeline(&self) -> &str;
}

pub fn get_material_config(config_str: &str) -> Result<Box<dyn MaterialConfig>, MaterialConfigError> {
    #[derive(serde::Deserialize)]
    struct ConfigWrapper<T> {
        material: T,
    }

    #[derive(serde::Deserialize)]
    struct PipelineName {
        pipeline: ConfigID,
    }

    // Extract pipeline name.
    let name = toml::from_str::<ConfigWrapper<PipelineName>>(&config_str)?
        .material
        .pipeline;

    fn read_config<T: MaterialConfig + DeserializeOwned + 'static>(
        config_str: &str,
    ) -> Result<Box<dyn MaterialConfig>, MaterialConfigError> {
        Ok(Box::new(toml::from_str::<ConfigWrapper<T>>(&config_str)?.material))
    }

    macro_rules! match_and_read_config {
        ($($config:ty),*) => {
            match name.as_str() {
                $(<$config>::PIPELINE => read_config::<$config>(&config_str),)*
                _ => Err(MaterialConfigError::UnknownPipeline(name)),
            }
        }
    }

    // TODO: Generate this with a macro, or use a const hash table.
    match_and_read_config!(UnlitMaterialConfig, LambertianMaterialConfig)
}

// Unlit material.
#[derive(serde::Deserialize, Debug)]
pub struct UnlitMaterialConfig {
    texture: ConfigID,
}

impl UnlitMaterialConfig {
    const PIPELINE: &'static str = "simple/unlit";
}

impl MaterialConfig for UnlitMaterialConfig {
    fn pipeline(&self) -> &str {
        Self::PIPELINE
    }
}

// Lambertian material.
#[derive(serde::Deserialize, Debug)]
pub struct LambertianMaterialConfig {
    albedo: ConfigID,
}

impl LambertianMaterialConfig {
    const PIPELINE: &'static str = "simple/lambertian";
}

impl MaterialConfig for LambertianMaterialConfig {
    fn pipeline(&self) -> &str {
        Self::PIPELINE
    }
}
*/
