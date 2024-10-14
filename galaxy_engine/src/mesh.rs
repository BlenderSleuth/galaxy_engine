// Copyright (c) 2024. Ben Sutherland

use std::alloc::Layout;
use std::fs::File;
use std::io::BufReader;
use std::slice;
use std::sync::Arc;

use ash::vk;
use meshopt::VertexDataAdapter;
use nalgebra as na;

use crate::buffer::{Buffer, CpuToGpu, GpuOnly};
use crate::command_buffer::CommandBuffer;
use crate::descriptors::DescriptorPool;
use crate::device::{Device, QueueFamily, SharedDeviceLoader};
use crate::gpu_alloc::{MemResult, MemoryError};
use crate::material::{Material, MaterialError};
use crate::pipeline::{GraphicsPipeline, GraphicsPipelineParameters, Pipeline, PipelineLayout};
use crate::uniform_buffer::VolatileUniformBuffer;
use crate::{debug, maths};

// For vertices with N attributes.
pub trait BindableVertex<const N: usize> {
    fn binding_description() -> vk::VertexInputBindingDescription;
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; N];
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pub position: na::Vector3<f32>,
    pub tex_coord: na::Vector2<f32>,
}

impl BindableVertex<2> for Vertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

pub trait IndexTypeTrait: bytemuck::Pod {
    fn index_type() -> vk::IndexType;
}
impl IndexTypeTrait for u16 {
    fn index_type() -> vk::IndexType {
        vk::IndexType::UINT16
    }
}
impl IndexTypeTrait for u32 {
    fn index_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
}

pub struct MeshBuffer {
    // Buffer contains vertices followed by indices.
    buffer: Buffer<GpuOnly>,
    num_indices: u32,
    index_type: vk::IndexType,
    vertices_offset: vk::DeviceSize,
}

impl MeshBuffer {
    pub fn new_from_vertices_and_indices<V: bytemuck::Pod, I: IndexTypeTrait>(
        name: &str,
        vertices: &[V],
        indices: &[I],
        device: &Device,
        transfer_cmd_pool: vk::CommandPool,
    ) -> MemResult<MeshBuffer> {
        // Ensure proper alignment.
        let indices_layout = Layout::for_value(indices);
        let vertices_offset = indices_layout
            .align_to(std::mem::align_of::<V>())
            .unwrap()
            .pad_to_align()
            .size();
        let buffer_size = (vertices_offset + std::mem::size_of_val(vertices)) as vk::DeviceSize;

        let mut buffer = Buffer::<GpuOnly>::new(
            debug::debug_only_name!("{name} mesh buffer"),
            &device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;

        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            debug::debug_only_name!("{name} mesh staging buffer"),
            &device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        staging_buffer.copy_into_buffer(indices, 0)?;
        staging_buffer.copy_into_buffer(vertices, vertices_offset)?;
        staging_buffer.copy_to_buffer(
            CommandBuffer::one_time_transient(device, transfer_cmd_pool)?,
            &device,
            &mut buffer,
            staging_buffer.size(),
            QueueFamily::Graphics,
        )?;

        Ok(Self {
            buffer,
            num_indices: indices.len() as u32,
            index_type: I::index_type(),
            vertices_offset: vertices_offset as vk::DeviceSize,
        })
    }

    pub fn num_indices(&self) -> u32 {
        self.num_indices
    }

    pub fn bind(&self, loader: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe { loader.cmd_bind_index_buffer(command_buffer, self.buffer.handle(), 0, self.index_type) };
        unsafe {
            loader.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                slice::from_ref(&self.buffer.handle()),
                &[self.vertices_offset],
            )
        };
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct ColouredVertex {
    pub position: na::Vector3<f32>,
    pub colour: na::Vector3<f32>,
    pub tex_coord: na::Vector2<f32>,
}

impl BindableVertex<3> for ColouredVertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ColouredVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, colour) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, tex_coord) as u32),
        ]
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Obj error: {0}")]
    ObjError(#[from] obj::ObjError),
    #[error("Mesh vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Material error: {0}")]
    MaterialError(#[from] MaterialError),
}

pub struct Mesh {
    loader: SharedDeviceLoader,
    mesh_buffer: MeshBuffer,
    material: Material,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline: GraphicsPipeline,
    pub mvp: maths::ModelViewProjection,
}

impl Mesh {
    pub fn new(
        name: &str,
        device: &Device,
        gfx_cmd_pool: vk::CommandPool,
        mesh_path: &str,
        texture_path: &str,
        samples: vk::SampleCountFlags,
        uniform_buffer: &VolatileUniformBuffer,
        descriptor_pool: &DescriptorPool,
    ) -> Result<Self, MeshError> {
        // Load material.
        let material = Material::new(
            debug::debug_only_name!("{name} material"),
            device,
            texture_path,
            gfx_cmd_pool,
        )?;

        // Load model. The obj crate already does indexing for us.
        let obj_model: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(BufReader::new(File::open(mesh_path)?))?;

        let vertices = obj_model
            .vertices
            .iter()
            .map(|v| Vertex {
                position: na::Vector3::new(v.position[0], v.position[1], v.position[2]),
                tex_coord: na::Vector2::new(v.texture[0], 1.0 - v.texture[1]),
            })
            .collect::<Vec<Vertex>>();

        // Optimize model.
        let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vertices, Some(&obj_model.indices));
        let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &vert_remap);
        let mut indices = meshopt::remap_index_buffer(Some(&obj_model.indices), vertex_count, &vert_remap);
        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);
        let vertex_data_adapter = VertexDataAdapter::new(
            bytemuck::must_cast_slice(&vertices),
            std::mem::size_of::<Vertex>(),
            std::mem::offset_of!(Vertex, position),
        )
        .unwrap();
        meshopt::optimize_overdraw_in_place(&mut indices, &vertex_data_adapter, 1.05);
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

        let mesh_buffer = MeshBuffer::new_from_vertices_and_indices(name, &vertices, &indices, device, gfx_cmd_pool)?;

        let layout_bindings = material.descriptor_set_layout_bindings();
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout = unsafe {
            device
                .loader()
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }?;

        // Create pipeline layout.
        let pipeline_layout = Arc::new(PipelineLayout::new(
            &device,
            Some(&descriptor_set_layout),
            Some(&maths::ModelViewProjection::push_constant_range()),
        )?);

        // Create pipeline.
        let pipeline_params = GraphicsPipelineParameters {
            layout: pipeline_layout,
            vertex_binding_description: Vertex::binding_description(),
            vertex_attribute_descriptions: &Vertex::attribute_descriptions(),
            shader_stages: material.shader_stages(),
            samples,
            depth_test: true,
        };
        let pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        // Create mesh descriptor sets.
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.handle())
            .set_layouts(slice::from_ref(&descriptor_set_layout));
        let descriptor_set = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?[0];

        let buffer_info = uniform_buffer.descriptor_buffer_info();

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(material.texture_image().view().handle())
            .sampler(material.sampler());

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(slice::from_ref(&image_info)),
        ];

        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };
        Ok(Self {
            loader: device.cloned_loader(),
            mesh_buffer,
            material,
            descriptor_set_layout,
            descriptor_set,
            pipeline,
            mvp: maths::ModelViewProjection::default(),
        })
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn record_graphics(
        &self,
        loader: &ash::Device,
        command_buffer: vk::CommandBuffer,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle()) };
        self.mesh_buffer.bind(loader, command_buffer);
        unsafe {
            loader.cmd_push_constants(
                command_buffer,
                self.pipeline.layout().handle(),
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&[self.mvp.mvp()]),
            )
        };
        unsafe {
            loader.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout().handle(),
                0,
                slice::from_ref(&self.descriptor_set),
                &[],
            )
        };
        unsafe { loader.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { loader.cmd_set_scissor(command_buffer, 0, &[scissor]) };
        unsafe { loader.cmd_draw_indexed(command_buffer, self.mesh_buffer.num_indices(), 1, 0, 0, 0) };
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
