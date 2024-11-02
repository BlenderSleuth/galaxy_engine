// Copyright (c) 2024 Ben Sutherland.

use std::slice;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;

use crate::vulkan::device::{Device, DeviceExt, SharedDeviceLoader};
use crate::vulkan::shader::{ComputeShaderStage, ShaderModule};

pub trait Pipeline {
    fn handle(&self) -> vk::Pipeline;
    fn layout(&self) -> &Arc<PipelineLayout>;
}

pub struct PipelineLayout {
    loader: SharedDeviceLoader,
    handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn new(
        device: &Device,
        descriptor_set_layout: Option<&vk::DescriptorSetLayout>,
        push_constant_range: Option<&vk::PushConstantRange>,
    ) -> VkResult<Self> {
        let loader = device.cloned_loader();

        let mut pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        if let Some(descriptor_set_layout) = descriptor_set_layout {
            pipeline_layout_info = pipeline_layout_info.set_layouts(slice::from_ref(descriptor_set_layout));
        }
        if let Some(push_constant_range) = push_constant_range {
            pipeline_layout_info = pipeline_layout_info.push_constant_ranges(slice::from_ref(&push_constant_range));
        }
        let handle = unsafe { loader.create_pipeline_layout(&pipeline_layout_info, None) }?;

        Ok(Self { loader, handle })
    }
    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_pipeline_layout(self.handle, None) }
    }
}

// Vertex, Hull, Domain, Geometry, Fragment.
pub const MAX_GRAPHICS_SHADER_STAGES: usize = 5;
pub type GraphicsPipelineShaderStages<'a> = ArrayVec<vk::PipelineShaderStageCreateInfo<'a>, MAX_GRAPHICS_SHADER_STAGES>;

pub struct GraphicsPipelineParameters<'a> {
    pub layout: Arc<PipelineLayout>,
    pub vertex_binding_description: vk::VertexInputBindingDescription,
    pub vertex_attribute_descriptions: &'a [vk::VertexInputAttributeDescription],
    pub shader_stages: GraphicsPipelineShaderStages<'a>,
    pub samples: vk::SampleCountFlags,
    pub depth_test: bool,
}

pub struct GraphicsPipeline {
    loader: SharedDeviceLoader,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

impl GraphicsPipeline {
    pub fn new(device: &Device, params: GraphicsPipelineParameters) -> VkResult<Self> {
        let loader = device.cloned_loader();

        let device_properties = device.physical_device();

        // Create graphics pipeline.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let pipeline_dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false) // Sample shading adds extra samples to
            .rasterization_samples(params.samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(slice::from_ref(&color_blend_attachment));

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(params.depth_test)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default());

        let mut dynamic_pipeline_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&device_properties.swapchain_format.format))
            .depth_attachment_format(device.physical_device().depth_stencil_format);

        // Vertex binding.
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&params.vertex_binding_description))
            .vertex_attribute_descriptions(&params.vertex_attribute_descriptions);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&params.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&pipeline_dynamic_state)
            .layout(params.layout.handle())
            .push_next(&mut dynamic_pipeline_info);

        let handle = unsafe { loader.create_graphics_pipeline(vk::PipelineCache::null(), &pipeline_info, None) }?;

        Ok(Self {
            loader,
            handle,
            layout: params.layout,
        })
    }
}

impl Pipeline for GraphicsPipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_pipeline(self.handle, None) }
    }
}

pub struct ComputePipelineParameters {
    pub layout: Arc<PipelineLayout>,
    pub compute_module: ShaderModule<ComputeShaderStage>,
}

pub struct ComputePipeline {
    loader: SharedDeviceLoader,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

impl ComputePipeline {
    pub fn new(device: &Device, params: ComputePipelineParameters) -> VkResult<Self> {
        let loader = device.cloned_loader();

        let compute_stage = params.compute_module.stage_info();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(compute_stage)
            .layout(params.layout.handle());

        // Non allocating version of create_graphics_pipelines.
        let mut handle = vk::Pipeline::null();
        let err_code = unsafe {
            (loader.fp_v1_0().create_compute_pipelines)(
                loader.handle(),
                vk::PipelineCache::null(),
                1,
                &pipeline_info,
                core::ptr::null(),
                &mut handle,
            )
        };
        if err_code != vk::Result::SUCCESS {
            return Err(err_code);
        }
        Ok(Self {
            loader,
            handle,
            layout: params.layout,
        })
    }
}

impl Pipeline for ComputePipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_pipeline(self.handle, None) }
    }
}
