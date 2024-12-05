// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::engine::GalaxyEngine;
use crate::level::DrawData;
use crate::materials::config::{get_material_config, ResourceBinding};
use crate::pipelines::{GraphicsPipeline, Pipeline, PipelineLayout};
use crate::resource_paths::{resource_type, ResourcePath};
use crate::textures::{TextureError, TextureIndex, TextureManager};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState, TransientPrimaryCommandPool};
use crate::vulkan::gpu_alloc::MemoryError;
use crate::vulkan::queue::queue_type::PrimaryQueue;

#[derive(thiserror::Error, Debug)]
pub enum MaterialError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
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

pub enum MaterialResourceBinding {
    Texture(TextureIndex),
}

// To be kept up to date with shader representation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialData {
    pub texture_index: TextureIndex,
}

pub struct Material {
    pipeline: Arc<GraphicsPipeline>,
    resource_bindings: HashMap<String, MaterialResourceBinding>,
}

impl Material {
    pub fn new(
        engine: &GalaxyEngine,
        texture_manager: &TextureManager,
        resource_path: &ResourcePath,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, MaterialError> {
        // Load config.
        let config_str = std::fs::read_to_string(&resource_path.full_path::<resource_type::Material>(engine))?;
        let config = get_material_config(&config_str)?;

        let pipeline = engine
            .pipeline_manager
            .get_graphics_pipeline(config.pipeline)
            .ok_or(MaterialError::PipelineNotFound(config.pipeline.to_owned()))?;

        // Construct resource bindings.
        let mut resource_bindings = HashMap::new();
        for (bind_point, binding) in config.params {
            match binding {
                ResourceBinding::Texture(path) => {
                    // Load texture.
                    //let texture_path = resource_path.relative_resource(path);
                    let texture_path = ResourcePath::new(&path, Some(resource_path))
                        .ok_or(MaterialError::ResourceError(path.to_owned()))?;
                    let texture_index = texture_manager.load_texture(path, &texture_path, engine, cmd_pool)?;
                    // Add to resource bindings.
                    resource_bindings.insert(bind_point.to_owned(), MaterialResourceBinding::Texture(texture_index));
                }
                _ => {
                    unimplemented!("Material binding not implemented");
                }
            }
        }

        Ok(Self {
            pipeline,
            resource_bindings,
        })
    }

    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline.layout()
    }

    pub fn resource_binding(&self, bind_point: &str) -> Option<&MaterialResourceBinding> {
        self.resource_bindings.get(bind_point)
    }

    pub fn bind(&self, cmd_buf: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buf.bind_graphics_pipeline(&self.pipeline);
        cmd_buf.push_constants(
            self.pipeline_layout(),
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&DrawData {
                transform_index: 0,
                material_index: 0,
            }),
        );
    }

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
