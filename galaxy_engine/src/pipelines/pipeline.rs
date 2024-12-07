// Copyright (c) 2024 Ben Sutherland.

use std::slice;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use ash::vk::Handle;
use itertools::izip;

use crate::pipelines::config::{GraphicsPipelineConfig, GraphicsShaderStageFlags, PipelineBinding, PipelineBindingMap};
use crate::pipelines::pipeline_manager::{FragmentShaderModuleCache, PipelineLayoutCache, VertexShaderModuleCache};
use crate::pipelines::{PipelineBindingDataSize, PipelineLayout};
use crate::utils::CStructLayout;
use crate::vertex_input::{BindableVertex, PositionColourTexCoordVertex, PositionTexCoordVertex, VertexInputType};
use crate::vulkan::device::Device;
use crate::vulkan::shader::{ComputeShaderStage, ShaderModule};

//#[derive(serde::Deserialize, Debug)]
//#[serde(rename_all = "lowercase")]
//pub enum PipelineType {
//    Graphics,
//    Compute,
//}

//impl PipelineType {
//    pub fn bind_point(&self) -> vk::PipelineBindPoint {
//        match self {
//            Self::Graphics => vk::PipelineBindPoint::GRAPHICS,
//            Self::Compute => vk::PipelineBindPoint::COMPUTE,
//        }
//    }
//}

pub trait Pipeline {
    fn handle(&self) -> vk::Pipeline;
    fn name(&self) -> &str;
    fn cloned_name(&self) -> Arc<str>;
    fn layout(&self) -> &PipelineLayout;
    fn bindings(&self) -> &PipelineBindingMap;
    fn bindings_layout(&self) -> CStructLayout {
        // Flags binding goes on the end.
        const FLAGS_BINDING: PipelineBinding = PipelineBinding {
            ty: PipelineBindingDataSize::Float,
            stages: GraphicsShaderStageFlags::all(),
        };
        CStructLayout::new(
            self.bindings()
                .values()
                .chain(std::iter::once(&FLAGS_BINDING))
                .map(|binding| binding.ty.layout()),
        )
        .unwrap()
    }
}

// Vertex, /*Hull, Domain, Geometry,*/ Fragment.
//pub const MAX_GRAPHICS_SHADER_STAGES: usize = 2;
//pub type GraphicsPipelineShaderStages =
//    ArrayVec<vk::PipelineShaderStageCreateInfo<'static>, MAX_GRAPHICS_SHADER_STAGES>;

//pub struct GraphicsPipelineParameters {
//    pub layout: Arc<PipelineLayout>,
//    pub vertex_binding_description: vk::VertexInputBindingDescription,
//    pub vertex_attribute_descriptions: &'static [vk::VertexInputAttributeDescription],
//    pub shader_stages: GraphicsPipelineShaderStages,
//    pub msaa_samples: vk::SampleCountFlags,
//    pub depth_test: bool,
//}

pub struct GraphicsPipeline {
    handle: vk::Pipeline,
    name: Arc<str>,
    layout: Arc<PipelineLayout>,
    bindings: PipelineBindingMap,
}

impl GraphicsPipeline {
    pub(super) fn batch_new(
        device: &Device,
        pipeline_layouts: &PipelineLayoutCache,
        vertex_shaders: VertexShaderModuleCache,
        fragment_shaders: FragmentShaderModuleCache,
        configs: Vec<GraphicsPipelineConfig>,
        msaa_samples: vk::SampleCountFlags,
    ) -> VkResult<Vec<Self>> {
        if configs.is_empty() {
            return Ok(Vec::new());
        }

        // Constant pipeline creation infos.
        let device_properties = device.physical_device();
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
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(slice::from_ref(&color_blend_attachment));
        let dynamic_pipeline_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&device_properties.swapchain_format.format))
            .depth_attachment_format(device.physical_device().depth_stencil_format);

        // Create shader stages arrays.
        let shader_stages: Vec<[vk::PipelineShaderStageCreateInfo; 2]> = configs
            .iter()
            .map(|config| {
                let vertex_shader_module = &vertex_shaders[&config.shaders.vertex.id];
                let fragment_shader_module = &fragment_shaders[&config.shaders.fragment.id];
                [vertex_shader_module.stage_info(), fragment_shader_module.stage_info()]
            })
            .collect();

        // Store all prerequisite create info for pipeline creation in a separate buffer (that can be referenced by the main one).
        let mut create_infos: Vec<_> = configs
            .iter()
            .map(|config| {
                (
                    // Vertex input state.
                    match config.shaders.vertex.input_type {
                        VertexInputType::PositionTexCoord => PositionTexCoordVertex::vertex_input_state(),
                        VertexInputType::PositionColourTexCoord => PositionColourTexCoordVertex::vertex_input_state(),
                    },
                    // Multisampling.
                    vk::PipelineMultisampleStateCreateInfo::default()
                        .sample_shading_enable(false)
                        .rasterization_samples(if config.rasteriser.multisample_enable {
                            msaa_samples
                        } else {
                            vk::SampleCountFlags::TYPE_1
                        }),
                    // Depth stencil.
                    vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(config.rasteriser.depth_enable)
                        .depth_write_enable(true)
                        .depth_compare_op(vk::CompareOp::GREATER)
                        .stencil_test_enable(false)
                        .front(Default::default())
                        .back(Default::default()),
                    dynamic_pipeline_info.clone(),
                )
            })
            .collect();

        let (pipeline_infos, pipeline_layouts): (Vec<_>, Vec<_>) = izip!(&configs, &shader_stages, &mut create_infos)
            .map(|(config, shader_stages, infos)| {
                let pipeline_layout = &pipeline_layouts[&config.layout.push_constant];
                (
                    vk::GraphicsPipelineCreateInfo::default()
                        .stages(shader_stages)
                        .vertex_input_state(&infos.0)
                        .input_assembly_state(&input_assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterizer)
                        .multisample_state(&infos.1)
                        .color_blend_state(&color_blend_state)
                        .depth_stencil_state(&infos.2)
                        .dynamic_state(&pipeline_dynamic_state)
                        .layout(pipeline_layout.handle())
                        .push_next(&mut infos.3),
                    pipeline_layout,
                )
            })
            .unzip();

        // Batch create graphics pipelines.
        let handles = match unsafe {
            device
                .loader()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        } {
            Ok(handles) => Ok(handles),
            Err((handles, err_code)) => {
                // Clean up any successfully created pipelines.
                for handle in handles {
                    if !handle.is_null() {
                        unsafe { device.loader().destroy_pipeline(handle, None) };
                    }
                }
                return Err(err_code);
            }
        }?;

        Ok(izip!(handles.into_iter(), pipeline_layouts, configs)
            .map(|(handle, layout, config)| Self {
                handle,
                name: config.name,
                layout: Arc::clone(layout),
                bindings: config.layout.bindings,
            })
            .collect())
    }
}

impl Pipeline for GraphicsPipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn cloned_name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }
    fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
    fn bindings(&self) -> &PipelineBindingMap {
        &self.bindings
    }
}

pub struct ComputePipelineParameters {
    pub layout: Arc<PipelineLayout>,
    pub name: Arc<str>,
    pub compute_module: ShaderModule<ComputeShaderStage>,
}

pub struct ComputePipeline {
    handle: vk::Pipeline,
    name: Arc<str>,
    layout: Arc<PipelineLayout>,
    bindings: PipelineBindingMap,
}

impl ComputePipeline {
    pub(super) fn new(device: &Device, params: ComputePipelineParameters) -> VkResult<Self> {
        let loader = device.cloned_loader();

        let compute_stage = params.compute_module.stage_info();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(compute_stage)
            .layout(params.layout.handle());

        // Non allocating version of create_compute_pipelines.
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
            handle,
            name: params.name,
            layout: params.layout,
            bindings: PipelineBindingMap::new(),
        })
    }
}

impl Pipeline for ComputePipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn cloned_name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }
    fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
    fn bindings(&self) -> &PipelineBindingMap {
        &self.bindings
    }
}
