// Copyright (c) 2024 Ben Sutherland.

use std::sync::Arc;

use ash::vk;
use ultraviolet::Vec4;
use crate::materials::config::{get_material_config, MaterialConfigError};
use crate::utils;
use crate::vertex_input::{BindableVertex, PositionTexCoordVertex};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState};
use crate::vulkan::descriptors::DescriptorSetLayout;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemoryError;
use crate::vulkan::pipeline::{GraphicsPipeline, GraphicsPipelineParameters, Pipeline, PipelineLayout};
use crate::vulkan::queue::queue_type::PrimaryQueue;
use crate::vulkan::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};

#[derive(thiserror::Error, Debug)]
pub enum MaterialError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Config error: {0}")]
    ConfigError(#[from] MaterialConfigError),
    #[error("Material vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Material memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

pub struct Material {
    vertex_shader_module: ShaderModule<VertexShaderStage>,
    fragment_shader_module: ShaderModule<FragmentShaderStage>,
    pipeline: GraphicsPipeline,
}

// To be kept up to date with shader representation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialData {
    pub albedo: Vec4,
    pub texture_index: u32,
}

impl Material {
    pub fn new(
        device: &Device,
        filepath: &str,
        descriptor_set_layout: &DescriptorSetLayout,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, MaterialError> {
        // Load config.
        let config_str = std::fs::read_to_string(filepath)?;
        let config = get_material_config(&config_str)?;
        println!("{:?}", config.pipeline());

        // Load shaders.
        let vertex_shader_module = ShaderModule::new(&device, "galaxy_engine/content/shaders/common/apply_mvp_vs.spv")?;
        let fragment_shader_module = ShaderModule::new(&device, "galaxy_engine/content/shaders/simple/unlit.spv")?;

        let shader_stages =
            utils::arrayvec_from_array([vertex_shader_module.stage_info(), fragment_shader_module.stage_info()]);

        // Create pipeline layout.
        let pipeline_layout = Arc::new(PipelineLayout::new(
            &device,
            Some(&[descriptor_set_layout.handle()]),
            None,
        )?);

        // Create pipeline.
        let pipeline_params = GraphicsPipelineParameters {
            layout: pipeline_layout,
            vertex_binding_description: PositionTexCoordVertex::binding_description(),
            vertex_attribute_descriptions: &PositionTexCoordVertex::attribute_descriptions(),
            shader_stages,
            samples,
            depth_test: true,
        };
        let pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        Ok(Self {
            vertex_shader_module,
            fragment_shader_module,
            pipeline,
        })
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
