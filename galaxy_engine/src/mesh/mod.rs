// Copyright (c) 2024 Ben Sutherland.

pub mod mesh_manager;

use std::alloc::Layout;
use std::fs::File;
use std::io::BufReader;

use ash::vk;
use meshopt::VertexDataAdapter;

use crate::engine::GalaxyEngine;
use crate::prelude::*;
use crate::resources::MeshResourcePath;
use crate::vertex_input::PositionTexCoordVertex;
use crate::vulkan::buffer::{Buffer, CpuToGpu, GpuOnly};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingCmdBuf, RenderingState, TransientPrimaryCommandPool};
use crate::vulkan::debug;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::queue::queue_type::PrimaryQueue;

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
    mesh_buffer: MeshBuffer,
}

impl Mesh {
    pub fn new(
        name: &str,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        mesh_path: &MeshResourcePath,
    ) -> Result<Self, MeshError> {
        // Load model. The obj crate already does indexing for us.
        let obj_model: obj::Obj<obj::TexturedVertex, u32> =
            obj::load_obj(BufReader::new(File::open(mesh_path.full_path(engine))?))?;

        let vertices = obj_model
            .vertices
            .iter()
            .map(|v| PositionTexCoordVertex {
                position: Vec3::new(v.position[0], v.position[1], v.position[2]),
                tex_coord: Vec2::new(v.texture[0], 1.0 - v.texture[1]),
            })
            .collect::<Vec<PositionTexCoordVertex>>();

        // Optimize model.
        let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vertices, Some(&obj_model.indices));
        let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &vert_remap);
        let mut indices = meshopt::remap_index_buffer(Some(&obj_model.indices), vertex_count, &vert_remap);
        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);
        let vertex_data_adapter = VertexDataAdapter::new(
            bytemuck::must_cast_slice(&vertices),
            std::mem::size_of::<PositionTexCoordVertex>(),
            std::mem::offset_of!(PositionTexCoordVertex, position),
        )
        .unwrap();
        meshopt::optimize_overdraw_in_place(&mut indices, &vertex_data_adapter, 1.05);
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

        let mesh_buffer =
            MeshBuffer::new_from_vertices_and_indices(name, &vertices, &indices, &engine.device, cmd_pool)?;

        Ok(Self { mesh_buffer })
    }

    pub fn bind(&self, cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>) {
        self.mesh_buffer.bind(cmd_buf);
    }

    pub fn draw(&self, cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>) {
        cmd_buf.draw_indexed(self.mesh_buffer.num_indices(), 1, 0, 0, 0);
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
