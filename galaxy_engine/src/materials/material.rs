// Copyright (c) 2024 Ben Sutherland.

use std::sync::Arc;

use ash::vk;

use crate::materials::config::get_material_config;
use crate::pipelines::{GraphicsPipeline, Pipeline, PipelineLayout, PipelineManager};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState};
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
    #[error("Material pipeline not found")]
    PipelineNotFound,
    #[error("Texture error: {0}")]
    TextureError(#[from] crate::texture::TextureError),
}

// To be kept up to date with shader representation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialData {
    pub texture_index: u32,
}

pub struct Material {
    pipeline: Arc<GraphicsPipeline>,
}

impl Material {
    pub fn new(pipeline_manager: &PipelineManager, config_path: &str) -> Result<Self, MaterialError> {
        // Load config.
        let config_str = std::fs::read_to_string(config_path)?;
        let config = get_material_config(&config_str)?;

        let pipeline = pipeline_manager
            .get_graphics_pipeline(&config.pipeline)
            .ok_or(MaterialError::PipelineNotFound)?;

        //for (bind_point, binding) in config.params {}

        Ok(Self { pipeline })
    }

    pub fn bind(&self, cmd_buf: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buf.bind_graphics_pipeline(&self.pipeline);
    }

    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline.layout()
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
