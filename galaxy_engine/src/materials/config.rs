// Copyright (c) 2024 Ben Sutherland.

use indexmap::IndexMap;

//pub enum MaterialLayoutBinding {
//    Constant(ConfigID),
//    Texture(ConfigID),
//}

#[derive(serde::Deserialize, Debug)]
#[serde(rename = "Material")]
pub(crate) struct MaterialConfig {
    pub pipeline: String,
    // TODO: based on the pipeline, use a struct serialiser.
    pub params: IndexMap<String, String>,
}

pub fn get_material_config(config_str: &str) -> ron::error::SpannedResult<MaterialConfig> {
    ron::from_str::<MaterialConfig>(&config_str)
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
