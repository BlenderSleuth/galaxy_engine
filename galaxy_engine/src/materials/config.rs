// Copyright (c) 2024 Ben Sutherland.

use serde::de::DeserializeOwned;
use serde::Deserialize;
use smol_str::SmolStr;

pub trait MaterialConfig: std::fmt::Debug {
    fn pipeline(&self) -> &str;
}

#[derive(thiserror::Error, Debug)]
pub enum MaterialConfigError {
    #[error("Unknown pipeline: {0}")]
    UnknownPipeline(SmolStr),
    #[error("TOML parse error: {0}")]
    TomlError(#[from] toml::de::Error),
}

pub fn get_material_config(config_str: &str) -> Result<Box<dyn MaterialConfig>, MaterialConfigError> {
    #[derive(Deserialize)]
    struct ConfigWrapper<T> {
        material: T,
    }

    #[derive(Deserialize)]
    struct PipelineName {
        pipeline: SmolStr,
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
#[derive(Deserialize, Debug)]
pub struct UnlitMaterialConfig {
    texture: SmolStr,
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
#[derive(Deserialize, Debug)]
pub struct LambertianMaterialConfig {
    albedo: SmolStr,
}

impl LambertianMaterialConfig {
    const PIPELINE: &'static str = "simple/lambertian";
}

impl MaterialConfig for LambertianMaterialConfig {
    fn pipeline(&self) -> &str {
        Self::PIPELINE
    }
}

