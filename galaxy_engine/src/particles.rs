// Copyright (c) 2024 Ben Sutherland.

use std::sync::Arc;

use ash::vk;

use crate::camera::ViewInfo;
use crate::engine::GalaxyEngine;
use crate::meshes::MeshBuffer;
use crate::pipelines::{ComputePipeline, GraphicsPipeline, Pipeline};
use crate::prelude::*;
use crate::vertex_input::{binding_description_for_type, BindableVertex, MeshVertex};
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::{CommandPool, RecordingCmdBuf, RenderingCmdBuf, Transient};
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::queue::queue_type::{ComputeQueueType, PrimaryQueue, QueueType};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Particle {
    position: Vec3,
    age: f32,
    velocity: Vec3,
    radius: f32,
    colour: Vec4,
}

impl BindableVertex<2> for Particle {
    fn binding_description() -> &'static vk::VertexInputBindingDescription {
        static DESCRIPTION: vk::VertexInputBindingDescription = binding_description_for_type::<Particle>();
        &DESCRIPTION
    }
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; 2] {
        static DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(Particle, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: std::mem::offset_of!(Particle, colour) as u32,
            },
        ];
        &DESCRIPTIONS
    }
}

pub struct GpuParticleSystem {
    loader: SharedDeviceLoader,
    max_num_particles: u32,
    compute_pipeline: ComputePipeline,
    graphics_pipeline: GraphicsPipeline,
    _particle_storage_buffer: Buffer<GpuOnly>,
    //_particles_indirect_buffer: Buffer<GpuOnly>,
    _descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    compute_descriptor_set: vk::DescriptorSet,
    mesh_buffer: Arc<MeshBuffer<MeshVertex>>,
}

impl GpuParticleSystem {
    pub fn new<Q: QueueType>(
        _device: &Device,
        _samples: vk::SampleCountFlags,
        _max_num_particles: u32,
        _window_size: vk::Extent2D,
        _cmd_pool: &mut CommandPool<Q, Transient>,
    ) -> MemResult<Self> {
        Err(MemoryError::VulkanError(vk::Result::ERROR_UNKNOWN))

        /*
        // Set up particle system compute pipeline.
        let particle_shader_module = ShaderModule::new(&device, "galaxy_engine/shaders/particles.comp.spv")?;

        // Initial particle positions.
        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let initial_particles = (0..max_num_particles)
            .map(|_| {
                let r = 0.25 * fastrand::f32().sqrt();
                let theta = 2.0 * std::f32::consts::PI * fastrand::f32();
                let x = r * theta.cos() * window_aspect_ratio;
                let y = r * theta.sin();
                let position = r * Vec2::new(x, y);
                let velocity = position.normalized() * fastrand::f32() * 0.25;
                Particle {
                    position: Vec3::new(position.x, position.y, fastrand::f32()),
                    age: fastrand::f32() * 10.,
                    velocity: Vec3::new(velocity.x, velocity.y, 0.0),
                    radius: 0.01,
                    color: Vec4::new(fastrand::f32(), fastrand::f32(), fastrand::f32(), 1.0),
                }
            })
            .collect::<Vec<_>>();

        let mut particle_storage_buffer = Buffer::<GpuOnly>::new_for_slice(
            "Particle storage buffer",
            &device,
            &initial_particles,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::VERTEX_BUFFER,
            None,
        )?;

        let mut cmd_buffer = cmd_pool.allocate_transient_cmd_buffer()?;
        // Ensures buffer sticks around for copy operation.
        let _staging_buffer = particle_storage_buffer.copy_via_staging_buffer(
            &device,
            &mut cmd_buffer,
            bytemuck::must_cast_slice(&initial_particles),
        )?;
        let cmd_buffer = cmd_buffer.end()?.submit(&[], &[])?;

        // TODO: Don't allocate small buffers.
        //let num_particles_buffer = Buffer::<GpuOnly>::new_for_type::<pod::vk::DrawIndexedIndirectCommand>(
        //    "Num particles buffer",
        //    &device,
        //    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
        //    None,
        //)?;

        // Create descriptor pool.
        let descriptor_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];

        let descriptor_pool = DescriptorPool::new(&device, &descriptor_pool_sizes)?;

        // Create compute descriptor set layout.
        let compute_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(
                    vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ),
            //     vk::DescriptorSetLayoutBinding::default()
            //         .binding(2)
            //         .descriptor_count(1)
            //         .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            //         .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let compute_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&compute_layout_bindings);
        let compute_descriptor_set_layout =
            unsafe { device.loader().create_descriptor_set_layout(&compute_layout_info, None) }?;

        // Allocate compute descriptor sets.
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.handle())
            .set_layouts(slice::from_ref(&compute_descriptor_set_layout));
        let compute_descriptor_set = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?[0];

        // Write descriptor sets.
        let buffer_infos = [
            particle_storage_buffer.descriptor_buffer_info(),
            //num_particles_buffer.descriptor_buffer_info(),
        ];

        let descriptor_writes = [
            // Current frame's storage buffer.
            vk::WriteDescriptorSet::default()
                .dst_set(compute_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(&buffer_infos[0])),
            // Draw indirect buffer.
            //vk::WriteDescriptorSet::default()
            //    .dst_set(compute_descriptor_set)
            //    .dst_binding(1)
            //    .dst_array_element(0)
            //    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            //    .buffer_info(slice::from_ref(&buffer_infos[1])),
        ];

        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        let compute_pipeline_layout = Arc::new(PipelineLayout::new(
            &device,
            Some(&[compute_descriptor_set_layout]),
            None,
        )?);

        let compute_pipeline_params = ComputePipelineParameters {
            layout: Arc::clone(&compute_pipeline_layout),
            compute_module: particle_shader_module,
        };
        let compute_pipeline = ComputePipeline::new(&device, compute_pipeline_params)?;

        // Graphics pipeline.
        let vertex_shader_module =
            ShaderModule::<VertexShaderStage>::new(device, "galaxy_engine/shaders/particles.vert.spv")?;
        let fragment_shader_module =
            ShaderModule::<FragmentShaderStage>::new(device, "galaxy_engine/shaders/particles.frag.spv")?;
        let particle_shader_stages =
            utils::arrayvec_from_array([vertex_shader_module.stage_info(), fragment_shader_module.stage_info()]);

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<Mat4>() as u32);
        let graphics_pipeline_layout = Arc::new(PipelineLayout::new(
            device,
            Some(&[compute_descriptor_set_layout]),
            Some(&push_constant_range),
        )?);

        let pipeline_params = GraphicsPipelineParameters {
            layout: graphics_pipeline_layout,
            vertex_binding_description: PositionTexCoordVertex::binding_description(),
            vertex_attribute_descriptions: &PositionTexCoordVertex::attribute_descriptions(),
            shader_stages: particle_shader_stages,
            samples,
            depth_test: true,
        };
        let graphics_pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        cmd_buffer.wait_for_fence()?;

        Ok(Self {
            loader: device.cloned_loader(),
            max_num_particles,
            compute_pipeline,
            graphics_pipeline,
            _particle_storage_buffer: particle_storage_buffer,
            //_particles_indirect_buffer: num_particles_buffer,
            _descriptor_pool: descriptor_pool,
            compute_descriptor_set_layout,
            compute_descriptor_set,
            mesh_buffer: engine::static_resources().get_octagon_cloned(),
        })*/
    }

    pub fn record_compute(&self, cmd_buffer: &mut RecordingCmdBuf<impl ComputeQueueType>) {
        cmd_buffer.bind_compute_pipeline(&self.compute_pipeline);
        cmd_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipeline.layout(),
            0,
            &[self.compute_descriptor_set],
            &[],
        );
        cmd_buffer.dispatch(self.max_num_particles / 256, 1, 1);
    }

    pub fn record_graphics(
        &self,
        command_buffer: &mut RenderingCmdBuf<PrimaryQueue>,
        view_info: &ViewInfo,
        time: f32,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        let mvp = view_info.mvp_from_similarity(&spin_transform(time.sin() * 0.5, 20.0));
        let pipeline_layout = self.graphics_pipeline.layout();
        command_buffer.bind_graphics_pipeline(&self.graphics_pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[self.compute_descriptor_set],
            &[],
        );
        command_buffer.push_constants(
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::bytes_of(&mvp),
        );
        //self.mesh_buffer.bind(command_buffer);
        command_buffer.set_viewport(viewport);
        command_buffer.set_scissor(scissor);
        command_buffer.draw_indexed(self.mesh_buffer.num_indices(), self.max_num_particles, 0, 0, 0);
    }
}

impl Drop for GpuParticleSystem {
    fn drop(&mut self) {
        // Drop descriptor set layouts.
        unsafe {
            self.loader
                .destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None)
        };
    }
}
