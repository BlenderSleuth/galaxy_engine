// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use ultraviolet::{Vec2, Vec3, Vec4};

use crate::engine::GalaxyEngine;
use crate::materials::config::{MaterialConfig, MaterialConfigError, ResourceBindingConfig};
use crate::pipelines::{GraphicsPipeline, Pipeline};
use crate::resource_paths::ResourcePath;
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

#[derive(serde::Serialize, serde::Deserialize, Debug, Copy, Clone)]
pub enum ResourceConstant {
    RGB(u8, u8, u8),
}

impl ResourceConstant {
    pub fn as_f32(&self) -> f32 {
        match *self {
            ResourceConstant::RGB(r, g, b) => (r as f32 + g as f32 + b as f32) / 3.0 / 255.0,
        }
    }
    pub fn as_vec2(&self) -> Vec2 {
        match *self {
            ResourceConstant::RGB(r, g, b) => Vec2::new(r as f32 / 255.0, g as f32 / 255.0),
        }
    }
    pub fn as_vec3(&self) -> Vec3 {
        match *self {
            ResourceConstant::RGB(r, g, b) => Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0),
        }
    }
    pub fn as_vec4(&self) -> Vec4 {
        match self {
            ResourceConstant::RGB(_, _, _) => self.as_vec3().into_homogeneous_point(),
        }
    }
}

pub enum ResourceBinding {
    Texture(u32),
    Constant(ResourceConstant),
}

pub type ResourceBindingMap = HashMap<Arc<str>, ResourceBinding>;

pub struct Material {
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
        resource_path: &ResourcePath,
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
                ResourceBindingConfig::Texture(path) => {
                    // Load texture.
                    let texture_path = ResourcePath::new(path, Some(resource_path))
                        .ok_or(MaterialError::ResourceError(path.to_owned()))?;
                    let texture_index = texture_manager.load_texture(path, &texture_path, engine, cmd_pool)?;
                    // Add to resource bindings.
                    resource_bindings.insert(id, ResourceBinding::Texture(texture_index));
                }
                ResourceBindingConfig::Constant(constant) => {
                    resource_bindings.insert(id, ResourceBinding::Constant(constant));
                }
            }
        }

        Ok(Self {
            pipeline,
            resource_bindings,
            level_index,
            //buffer_index,
        })
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

    pub fn resource_bindings(&self) -> &ResourceBindingMap {
        &self.resource_bindings
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

    //pub fn buffer_index(&self) -> u32 {
    //    self.buffer_index
    //}

    // pub fn shader_stages(&self) -> GraphicsPipelineShaderStages {
    //     utils::arrayvec_from_array([
    //         self.vertex_shader_module.stage_info(),
    //         self.fragment_shader_module.stage_info(),
    //     ])
    // }

    // pub fn texture_image(&self) -> &Image {
    //     &self.texture_image.image()
    // }

    // pub fn sampler(&self) -> vk::Sampler {
    //     self.texture_image.sampler()
    // }

    // pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
    //     self.descriptor_set_layout
    // }

    // pub fn descriptor_set_layout_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {}
}
