use crate::buffer::{Buffer, CpuToGpu, GpuOnly};
use crate::command_buffer::CommandBuffer;
use crate::device::{Device, QueueFamily, SharedDeviceLoader};
use crate::engine::GalaxyEngine;
use crate::gpu_alloc::MemResult;
use crate::pipeline::{ComputePipeline, ComputePipelineParameters, GraphicsPipeline, GraphicsPipelineParameters, Pipeline, PipelineLayout};
use crate::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};
use crate::uniform_buffer::VolatileUniformBuffer;
use crate::utils;
use ash::vk;
use nalgebra as na;
use std::slice;
use std::sync::Arc;
use crate::descriptors::DescriptorPool;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Particle {
    position: na::Vector2<f32>,
    velocity: na::Vector2<f32>,
    color: na::Vector4<f32>,
}

impl Particle {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Particle, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(std::mem::offset_of!(Particle, color) as u32),
        ]
    }
}

pub struct ParticleSystem {
    loader: SharedDeviceLoader,
    num_particles: u32,
    compute_pipeline: ComputePipeline,
    graphics_pipeline: GraphicsPipeline,
    particle_storage_buffers: Vec<Buffer<GpuOnly>>,
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    compute_descriptor_sets: [vk::DescriptorSet; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
}

impl ParticleSystem {
    pub fn new(
        device: &Device,
        samples: vk::SampleCountFlags,
        num_particles: u32,
        window_size: vk::Extent2D,
        uniform_buffer: &VolatileUniformBuffer,
        graphics_cmd_pool: vk::CommandPool,
        descriptor_pool: &DescriptorPool,
    ) -> MemResult<Self> {
        // Set up particle system compute pipeline.
        let particle_shader_code = std::fs::read("shaders/particles.comp.spv").unwrap();
        let particle_shader_module = ShaderModule::new(&device, &particle_shader_code)?;

        // Initial particle positions.
        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let initial_particles = (0..num_particles).map(|_| {
            let r = 0.25 * fastrand::f32().sqrt();
            let theta = 2.0 * std::f32::consts::PI * fastrand::f32();
            let x = r * theta.cos() * window_aspect_ratio;
            let y = r * theta.sin();
            let position = r * na::Vector2::new(x, y);
            Particle {
                position,
                velocity: position.normalize() * 0.25,
                color: na::Vector4::new(fastrand::f32(), fastrand::f32(), fastrand::f32(), 1.0),
            }
        }).collect::<Vec<_>>();

        // Copy to staging buffer.
        let mut particle_staging_buffer = Buffer::<CpuToGpu>::new(
            "Particle staging buffer",
            &device,
            1,
            std::mem::size_of::<Particle>() * num_particles as usize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        particle_staging_buffer.copy_into_buffer(bytemuck::cast_slice(&initial_particles), 0)?;

        let cmd_buffer = CommandBuffer::begin_one_time(&device, graphics_cmd_pool)?;
        let shader_storage_buffers = (0..GalaxyEngine::MAX_FRAMES_IN_FLIGHT).map(|_| {
            let mut buffer = Buffer::<GpuOnly>::new_for_typed_data(
                "Particle storage buffer",
                &device,
                &initial_particles,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::SharingMode::EXCLUSIVE,
            )?;
            let buffer_size = buffer.size();
            particle_staging_buffer.copy_to_buffer(cmd_buffer.as_persistent(), &device, &mut buffer, buffer_size, QueueFamily::Graphics)?;
            Ok(buffer)
        }).collect::<MemResult<Vec<_>>>()?;
        cmd_buffer.end_submit_and_wait(&device, device.get_queue(QueueFamily::Graphics))?;

        // Create compute descriptor set layout.
        let compute_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let compute_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&compute_layout_bindings);
        let compute_descriptor_set_layout = unsafe { device.loader().create_descriptor_set_layout(&compute_layout_info, None) }?;

        // Allocate compute descriptor sets.
        let layouts = [compute_descriptor_set_layout; GalaxyEngine::MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.handle())
            .set_layouts(&layouts);
        let compute_descriptor_sets = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?;

        // Write descriptor sets.
        let descriptor_buffer_infos = (0..GalaxyEngine::MAX_FRAMES_IN_FLIGHT).map(|i| {
            [
                uniform_buffer.descriptor_buffer_info(),
                vk::DescriptorBufferInfo::default()
                    .buffer(shader_storage_buffers[(i + 1) % GalaxyEngine::MAX_FRAMES_IN_FLIGHT].handle())
                    .offset(0)
                    .range((std::mem::size_of::<Particle>() * GalaxyEngine::NUM_PARTICLES as usize) as vk::DeviceSize),
                vk::DescriptorBufferInfo::default()
                    .buffer(shader_storage_buffers[i].handle())
                    .offset(0)
                    .range((std::mem::size_of::<Particle>() * GalaxyEngine::NUM_PARTICLES as usize) as vk::DeviceSize),
            ]
        }).collect::<Vec<_>>();

        let descriptor_writes = compute_descriptor_sets.iter().zip(descriptor_buffer_infos.iter()).flat_map(|(descriptor_set, buffer_infos)| {
            [
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[0])),
                // Last frame's storage buffer.
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[1])),
                // Current frame's storage buffer.
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[2])),
            ]
        }).collect::<Vec<_>>();
        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        let compute_pipeline_layout = Arc::new(PipelineLayout::new(&device, Some(&compute_descriptor_set_layout), None)?);

        let compute_pipeline_params = ComputePipelineParameters {
            layout: compute_pipeline_layout,
            compute_module: particle_shader_module,
        };
        let compute_pipeline = ComputePipeline::new(&device, compute_pipeline_params)?;

        // Graphics pipeline.
        let vertex_shader_code = std::fs::read("shaders/particles.vert.spv").unwrap();
        let fragment_shader_code = std::fs::read("shaders/particles.frag.spv").unwrap();
        let vertex_shader_module = ShaderModule::<VertexShaderStage>::new(device, &vertex_shader_code)?;
        let fragment_shader_module = ShaderModule::<FragmentShaderStage>::new(device, &fragment_shader_code)?;
        let particle_shader_stages = utils::arrayvec_from_array([
            vertex_shader_module.stage_info(),
            fragment_shader_module.stage_info(),
        ]);

        let pipeline_layout = PipelineLayout::new(device, None, None)?;

        let pipeline_params = GraphicsPipelineParameters {
            layout: Arc::new(pipeline_layout),
            vertex_binding_description: Particle::binding_description(),
            //vertex_binding_description: Vertex::binding_description(),
            vertex_attribute_descriptions: &Particle::attribute_descriptions(),
            //vertex_attribute_descriptions: &Vertex::attribute_descriptions(),
            shader_stages: particle_shader_stages,
            samples,
            depth_test: false,
            topology: vk::PrimitiveTopology::POINT_LIST,
        };
        let graphics_pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        Ok(Self {
            loader: device.cloned_loader(),
            num_particles,
            compute_pipeline,
            graphics_pipeline,
            particle_storage_buffers: shader_storage_buffers,
            compute_descriptor_set_layout,
            compute_descriptor_sets: compute_descriptor_sets.try_into().unwrap(),
        })
    }

    pub fn record_compute(&self, loader: &ash::Device, command_buffer: vk::CommandBuffer, current_frame: usize) {
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline.handle()) };
        unsafe { loader.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline.layout().handle(), 0, slice::from_ref(&self.compute_descriptor_sets[current_frame]), &[]) };
        unsafe { loader.cmd_dispatch(command_buffer, self.num_particles / 256, 1, 1) };
    }

    pub fn record_graphics(&self, loader: &ash::Device, command_buffer: vk::CommandBuffer, current_frame: usize, viewport: vk::Viewport, scissor: vk::Rect2D) {
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline.handle()) };
        unsafe { loader.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&self.particle_storage_buffers[current_frame].handle()), slice::from_ref(&0)) };
        unsafe { loader.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { loader.cmd_set_scissor(command_buffer, 0, &[scissor]) };
        unsafe { loader.cmd_draw(command_buffer, self.num_particles, 1, 0, 0) };
    }
}

impl Drop for ParticleSystem {
    fn drop(&mut self) {
        // Drop descriptor set layouts.
        unsafe { self.loader.destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None) };
    }
}