// Copyright (c) 2024 Ben Sutherland.

use std::alloc::Layout;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use ash::vk;
use meshopt::VertexDataAdapter;

use crate::material::Material;
use crate::prelude::*;
use crate::vulkan::buffer::{Buffer, CpuToGpu, GpuOnly};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingCmdBuf, RenderingState, TransientPrimaryCommandPool};
use crate::vulkan::debug;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::queue::queue_type::PrimaryQueue;

// For vertices with N attributes.
pub trait BindableVertex<const N: usize> {
    fn binding_description() -> vk::VertexInputBindingDescription;
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; N];
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pub position: Vec3,
    pub tex_coord: Vec2,
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
        cmd_pool: &mut TransientPrimaryCommandPool,
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
            None,
        )?;

        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            debug::debug_only_name!("{name} mesh staging buffer"),
            &device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            None,
        )?;
        staging_buffer.copy_slice_into_buffer(indices, 0)?;
        staging_buffer.copy_slice_into_buffer(vertices, vertices_offset)?;

        let mut cmd_buffer = cmd_pool.allocate_transient_cmd_buffer()?;
        staging_buffer.copy_to_buffer(&mut cmd_buffer, &mut buffer, staging_buffer.size());
        cmd_buffer.end_submit_wait_and_free()?;

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

    pub fn bind(&self, cmd_buffer: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buffer.bind_index_buffer(&self.buffer, 0, self.index_type);
        cmd_buffer.bind_vertex_buffer(&self.buffer, self.vertices_offset);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct ColouredVertex {
    pub position: Vec3,
    pub colour: Vec3,
    pub tex_coord: Vec2,
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
}

pub struct Mesh {
    loader: SharedDeviceLoader,
    mesh_buffer: MeshBuffer,
    material: Arc<Material>,
    pub transform: Similarity3,
}

impl Mesh {
    pub fn new(
        name: &str,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
        mesh_path: &str,
        material: Arc<Material>,
    ) -> Result<Self, MeshError> {
        // Load model. The obj crate already does indexing for us.
        let obj_model: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(BufReader::new(File::open(mesh_path)?))?;

        let vertices = obj_model
            .vertices
            .iter()
            .map(|v| Vertex {
                position: Vec3::new(v.position[0], v.position[1], v.position[2]),
                tex_coord: Vec2::new(v.texture[0], 1.0 - v.texture[1]),
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

        let mesh_buffer = MeshBuffer::new_from_vertices_and_indices(name, &vertices, &indices, device, cmd_pool)?;

        Ok(Self {
            loader: device.cloned_loader(),
            mesh_buffer,
            material,
            transform: Similarity3::identity(),
        })
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn record_graphics(&self, cmd_buffer: &mut RenderingCmdBuf<PrimaryQueue>) {
        self.mesh_buffer.bind(cmd_buffer);
        cmd_buffer.draw_indexed(self.mesh_buffer.num_indices(), 1, 0, 0, 0);
    }
}

//impl Drop for Mesh {
//    fn drop(&mut self) {
//        unsafe {
//            self.loader
//                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
//        }
//    }
//}
