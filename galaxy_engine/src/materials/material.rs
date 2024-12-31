// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::engine::GalaxyEngine;
use crate::materials::config::{MaterialConfig, MaterialConfigError, ResourceBindingConfig};
use crate::materials::ResourceBinding;
use crate::pipelines::{GraphicsPipeline, Pipeline};
use crate::resource_paths::{ResourcePath, SubresourcePath};
use crate::textures::{TextureError, TextureManager};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::gpu_alloc::MemoryError;

#[derive(thiserror::Error, Debug)]
pub enum MaterialError {
    #[error("Config error: {0}")]
    ConfigError(#[from] MaterialConfigError),
    #[error("Material vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Material memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Material pipeline not found: {0}")]
    PipelineNotFound(String),
    #[error("Texture error: {0}")]
    TextureError(#[from] TextureError),
    #[error("Resource error: {0}")]
    ResourceError(String),
}

type ResourceBindingMap = HashMap<Arc<str>, ResourceBinding>;

pub struct Material {
    path: SubresourcePath,
    pipeline: Arc<GraphicsPipeline>,
    resource_bindings: ResourceBindingMap,
    level_index: u32,
    //buffer_index: u32,
}

impl Material {
    pub(crate) fn new(
        engine: &GalaxyEngine,
        texture_manager: &mut TextureManager,
        config: &MaterialConfig,
        path: SubresourcePath,
        //buffer_index: u32,
        level_index: u32,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, MaterialError> {
        let pipeline = engine
            .pipeline_manager
            .get_cloned_graphics_pipeline(config.pipeline)
            .ok_or(MaterialError::PipelineNotFound(config.pipeline.to_owned()))?;

        // Construct resource bindings.
        let mut resource_bindings = HashMap::new();
        for (&bind_point, &binding) in config.params.iter() {
            let id = Arc::clone(
                engine
                    .pipeline_manager
                    .get_bind_point_id(bind_point)
                    .expect("Unknown pipeline bind point."),
            );
            match binding {
                ResourceBindingConfig::Texture(texture_path_str) => {
                    // Load texture.
                    let texture_path = ResourcePath::new(texture_path_str, Some(path.resource()))
                        .ok_or(MaterialError::ResourceError(texture_path_str.to_owned()))?;
                    let texture_index =
                        texture_manager.load_texture(texture_path_str, &texture_path, engine, cmd_pool)?;
                    // Add to resource bindings.
                    resource_bindings.insert(id, ResourceBinding::Texture(texture_index));
                }
                ResourceBindingConfig::Constant(constant) => {
                    resource_bindings.insert(id, ResourceBinding::Constant(constant));
                }
            }
        }

        Ok(Self {
            path,
            pipeline,
            resource_bindings,
            level_index,
            //buffer_index,
        })
    }

    pub fn path(&self) -> &SubresourcePath {
        &self.path
    }

    pub fn pipeline(&self) -> &GraphicsPipeline {
        &self.pipeline
    }

    pub fn cloned_pipeline(&self) -> Arc<GraphicsPipeline> {
        Arc::clone(&self.pipeline)
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline.layout()
    }

    pub fn get_resource_binding(&self, bind_point: &str) -> Option<&ResourceBinding> {
        self.resource_bindings.get(bind_point)
    }

    pub fn iter_resource_bindings(&self) -> impl Iterator<Item = &ResourceBinding> {
        self.resource_bindings.values()
    }

    pub fn resource_binding(&self, bind_point: &str) -> Option<&ResourceBinding> {
        self.resource_bindings.get(bind_point)
    }

    pub fn level_index(&self) -> u32 {
        self.level_index
    }
}
