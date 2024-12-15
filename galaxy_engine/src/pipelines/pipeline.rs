// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashSet;
use std::slice;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use ash::vk::Handle;
use itertools::izip;

use crate::pipelines::config::{
    ComputePipelineConfig, GraphicsPipelineBinding, GraphicsPipelineConfig, GraphicsShaderStageFlags,
    PipelineBindingMap,
};
use crate::pipelines::PipelineBindingDataSize;
use crate::utils::CStructLayout;
use crate::vertex_input::{BindableVertex, PositionColourTexCoordVertex, PositionTexCoordVertex, VertexInputType};
use crate::vulkan::device::Device;
use crate::vulkan::shader::{shader_stage, ShaderModule};

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

fn handle_pipeline_create(
    device: &Device,
    handles: Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) -> VkResult<Vec<vk::Pipeline>> {
    match handles {
        Ok(handles) => Ok(handles),
        Err((handles, err_code)) => {
            // Destroy any successfully created pipelines (unsuccessful are ignored).
            // All pipelines must be created for successful engine initialisation.
            for handle in handles {
                if !handle.is_null() {
                    unsafe { device.loader().destroy_pipeline(handle, None) };
                }
            }
            return Err(err_code);
        }
    }
}

pub trait Pipeline {
    fn handle(&self) -> vk::Pipeline;
    fn id(&self) -> &str;
    fn cloned_id(&self) -> Arc<str>;
    fn layout(&self) -> vk::PipelineLayout;
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

pub(crate) struct GraphicsPipelineCreateResources<'a> {
    pub config: GraphicsPipelineConfig<'a>,
    pub pipeline_layout: vk::PipelineLayout,
    pub vertex_shader: &'a ShaderModule<shader_stage::Vertex>,
    pub fragment_shader: &'a ShaderModule<shader_stage::Fragment>,
}

pub struct GraphicsPipeline {
    handle: vk::Pipeline,
    id: Arc<str>,
    layout: vk::PipelineLayout,
    bindings: PipelineBindingMap,
}

impl GraphicsPipeline {
    pub(super) fn batch_new(
        device: &Device,
        create_resources: Vec<GraphicsPipelineCreateResources>,
        msaa_samples: vk::SampleCountFlags,
        bind_point_id_cache: &mut HashSet<Arc<str>>,
    ) -> VkResult<Vec<Self>> {
        if create_resources.is_empty() {
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
        let shader_stages: Vec<[vk::PipelineShaderStageCreateInfo; 2]> = create_resources
            .iter()
            .map(|resources| {
                [
                    resources.vertex_shader.stage_info(),
                    resources.fragment_shader.stage_info(),
                ]
            })
            .collect();

        // Store all prerequisite create info for pipeline creation in a separate buffer (that can be referenced by the main one).
        let mut create_infos: Vec<_> = create_resources
            .iter()
            .map(|resources| {
                (
                    // Vertex input state.
                    match resources.config.shaders.vertex.input_type {
                        VertexInputType::PositionTexCoord => PositionTexCoordVertex::vertex_input_state(),
                        VertexInputType::PositionColourTexCoord => PositionColourTexCoordVertex::vertex_input_state(),
                    },
                    // Multisampling.
                    vk::PipelineMultisampleStateCreateInfo::default()
                        .sample_shading_enable(false)
                        .rasterization_samples(if resources.config.rasteriser.multisample_enable {
                            msaa_samples
                        } else {
                            vk::SampleCountFlags::TYPE_1
                        }),
                    // Depth stencil.
                    vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(resources.config.rasteriser.depth_enable)
                        .depth_write_enable(true)
                        .depth_compare_op(vk::CompareOp::GREATER)
                        .stencil_test_enable(false)
                        .front(Default::default())
                        .back(Default::default()),
                    dynamic_pipeline_info.clone(),
                )
            })
            .collect();

        let pipeline_infos = izip!(&create_resources, &shader_stages, &mut create_infos)
            .map(|(resources, shader_stages, infos)| {
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
                    .layout(resources.pipeline_layout)
                    .push_next(&mut infos.3)
            })
            .collect::<Vec<_>>();

        // Batch create graphics pipelines.
        let handles = handle_pipeline_create(device, unsafe {
            device
                .loader()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        })?;

        Ok(handles
            .into_iter()
            .zip(create_resources)
            .map(|(handle, resources)| Self {
                handle,
                id: resources.config.id,
                layout: resources.pipeline_layout,
                bindings: resources
                    .config
                    .layout
                    .bindings
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            // Until HashSet::get_or_insert is stabilized, we use an if-let.
                            if let Some(id) = bind_point_id_cache.get(k) {
                                Arc::clone(id)
                            } else {
                                let id = Arc::from(k);
                                bind_point_id_cache.insert(Arc::clone(&id));
                                id
                            },
                            v,
                        )
                    })
                    .collect(),
            })
            .collect())
    }
    pub fn bindings(&self) -> &PipelineBindingMap {
        &self.bindings
    }
    pub fn bindings_layout(&self) -> CStructLayout {
        // Flags binding goes on the end.
        const FLAGS_BINDING: GraphicsPipelineBinding = GraphicsPipelineBinding {
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

impl Pipeline for GraphicsPipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn cloned_id(&self) -> Arc<str> {
        Arc::clone(&self.id)
    }
    fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

pub(crate) struct ComputePipelineCreateResources<'a> {
    pub config: ComputePipelineConfig<'a>,
    pub pipeline_layout: vk::PipelineLayout,
    pub shader: &'a ShaderModule<shader_stage::Compute>,
}

pub struct ComputePipeline {
    handle: vk::Pipeline,
    id: Arc<str>,
    num_threads: [u32; 3],
    layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub(super) fn batch_new(
        device: &Device,
        create_resources: Vec<ComputePipelineCreateResources>,
    ) -> VkResult<Vec<Self>> {
        if create_resources.is_empty() {
            return Ok(Vec::new());
        }

        let pipeline_infos = create_resources
            .iter()
            .map(|resources| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(resources.shader.stage_info())
                    .layout(resources.pipeline_layout)
            })
            .collect::<Vec<_>>();

        // Batch create compute pipelines.
        let handles = handle_pipeline_create(device, unsafe {
            device
                .loader()
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        })?;

        Ok(handles
            .into_iter()
            .zip(create_resources)
            .map(|(handle, resources)| Self {
                handle,
                id: resources.config.id,
                num_threads: resources.config.num_threads,
                layout: resources.pipeline_layout,
            })
            .collect())
    }

    pub fn num_threads_per_group(&self) -> &[u32; 3] {
        &self.num_threads
    }

    pub fn total_threads_per_group(&self) -> u32 {
        self.num_threads.iter().product()
    }
}

impl Pipeline for ComputePipeline {
    fn handle(&self) -> vk::Pipeline {
        self.handle
    }
    fn id(&self) -> &str {
        &self.id
    }
    fn cloned_id(&self) -> Arc<str> {
        Arc::clone(&self.id)
    }
    fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}
