// Copyright (c) 2024-2025 Ben Sutherland.

mod load;
pub mod mesh_manager;

use std::alloc::Layout;

use ash::vk;

use crate::engine::GalaxyEngine;
use crate::loading::LoadingContext;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::vertex_input::MeshVertex;
use crate::vulkan::buffer::{Buffer, GpuOnly, Staging};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState};
use crate::vulkan::debug;
use crate::vulkan::device::queue::queue_type::PrimaryQueue;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::queue::QueueType;

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

pub struct MeshBuffer<V> {
    // Buffer contains vertices followed by indices.
    buffer: Buffer<GpuOnly>,
    num_vertices: u32,
    num_indices: u32,
    index_type: vk::IndexType,
    indices_offset: vk::DeviceSize,
    _vertex_format: std::marker::PhantomData<V>,
}

impl<V: bytemuck::Pod> MeshBuffer<V> {
    pub fn pad_vertices_size<I: IndexTypeTrait>(vertices_size: usize) -> usize {
        Layout::from_size_align(vertices_size, align_of::<I>())
            .unwrap()
            .pad_to_align()
            .size()
    }

    pub fn new_from_vertices_and_indices<I: IndexTypeTrait>(
        name: &str,
        vertices: &[V],
        indices: &[I],
        device: &Device,
        loading_ctx: &mut LoadingContext<impl QueueType>,
        usage: vk::BufferUsageFlags,
    ) -> MemResult<Self> {
        // Ensure proper alignment.
        let indices_offset = Self::pad_vertices_size::<I>(size_of_val(vertices));
        let buffer_size = (indices_offset + size_of_val(indices)) as vk::DeviceSize;

        let mut buffer = Buffer::<GpuOnly>::new(
            debug::debug_only_name!("{name} meshes buffer"),
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
        )?;

        loading_ctx.load(|mut cmd_buffer| {
            let mut staging_buffer = Buffer::<Staging>::new(
                debug::debug_only_name!("{name} meshes staging buffer"),
                device,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            staging_buffer.copy_slice_into_buffer(vertices, 0)?;
            staging_buffer.copy_slice_into_buffer(indices, indices_offset)?;

            staging_buffer.copy_to_buffer(&mut cmd_buffer, &mut buffer, staging_buffer.size());

            Ok([staging_buffer])
        })?;

        Ok(Self {
            buffer,
            num_vertices: vertices.len() as u32,
            num_indices: indices.len() as u32,
            index_type: I::index_type(),
            indices_offset: indices_offset as vk::DeviceSize,
            _vertex_format: std::marker::PhantomData,
        })
    }

    pub fn num_vertices(&self) -> u32 {
        self.num_vertices
    }

    pub fn num_indices(&self) -> u32 {
        self.num_indices
    }

    pub fn vertices_addr(&self) -> vk::DeviceAddress {
        self.buffer.device_address()
    }

    pub fn indices_addr(&self) -> vk::DeviceAddress {
        self.buffer.device_address() + self.indices_offset
    }

    pub fn bind(&self, cmd_buffer: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buffer.bind_vertex_buffer(&self.buffer, 0);
        cmd_buffer.bind_index_buffer(&self.buffer, self.indices_offset, self.index_type);
    }

    //pub fn draw(&self, cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>, first_index: u32, vertex_offset: i32) {
    //    cmd_buf.draw_indexed(self.num_indices(), 1, first_index, vertex_offset, 0);
    //}
}

struct MeshElementOffset {
    _vertex_offset: u32,
    vertex_count: u32,
    _index_offset: u32,
    index_count: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("Obj error: {0}")]
    ObjError(#[from] obj::ObjError),
    #[error("Mesh vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

pub struct Mesh {
    buffer: MeshBuffer<MeshVertex>,
    level_index: u32,          // Index of the mesh in the level.
    level_element_offset: u32, // Offset of the first element in the mesh in the level.
    elements: Vec<MeshElementOffset>,
}

impl Mesh {
    pub fn new(
        name: &str,
        engine: &GalaxyEngine,
        loading_ctx: &mut LoadingContext<impl QueueType>,
        mesh_path: &ResourcePath,
        level_index: u32,
        level_element_offset: u32,
    ) -> Result<Self, MeshError> {
        let obj_path = mesh_path.full_path::<resource_type::Mesh>(engine);
        let loaded_obj = load::MeshData::load_obj(&obj_path)?;
        let mesh_buffer = MeshBuffer::new_from_vertices_and_indices(
            name,
            &loaded_obj.vertices,
            &loaded_obj.indices,
            &engine.device,
            loading_ctx,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        log::info!(
            "Loaded mesh: {} has {} vertices and {} indices.",
            name,
            mesh_buffer.num_vertices(),
            mesh_buffer.num_indices()
        );
        Ok(Self {
            buffer: mesh_buffer,
            level_index,
            level_element_offset,
            elements: loaded_obj.elements,
        })
    }

    pub fn num_vertices(&self) -> u32 {
        self.buffer.num_vertices()
    }

    pub fn num_indices(&self) -> u32 {
        self.buffer.num_indices()
    }

    pub fn num_elements(&self) -> u32 {
        self.elements.len() as u32
    }

    //pub fn elements(&self) -> &[MeshElementOffset] {
    //    &self.elements
    //}

    pub fn level_index(&self) -> u32 {
        self.level_index
    }

    pub fn level_element_range(&self) -> std::ops::Range<usize> {
        let start = self.level_element_offset as usize;
        start..(start + self.elements.len())
    }

    pub fn buffer(&self) -> &MeshBuffer<MeshVertex> {
        &self.buffer
    }
}
