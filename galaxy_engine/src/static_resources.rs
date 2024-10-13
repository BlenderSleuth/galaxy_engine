use crate::buffer::{Buffer, GpuOnly};
use crate::device::{Device, QueueFamily};
use crate::gpu_alloc::MemResult;
use crate::mesh::{Vertex, VertexIndexBuffer};
use crate::utils;
use arrayvec::ArrayVec;
use ash::vk;
use nalgebra as na;
use std::ops::Deref;
use std::sync::Arc;

pub struct StaticResources {
    quad_buffer: Arc<VertexIndexBuffer>,
}

impl StaticResources {
    /// Initialisation of static resources.
    pub(crate) fn new(device: &Device, gfx_cmd_pool: vk::CommandPool) -> MemResult<Self> {
        Ok(Self {
            quad_buffer: Arc::new(Self::new_quad(device, gfx_cmd_pool)?),
        })
    }

    /// Quad vertex/index buffer.
    #[allow(dead_code)] // TODO: make library.
    pub fn get_quad(&self) -> &VertexIndexBuffer {
        self.quad_buffer.deref()
    }
    pub fn get_quad_cloned(&self) -> Arc<VertexIndexBuffer> {
        Arc::clone(&self.quad_buffer)
    }

    const QUAD_VERTICES: [Vertex; 4] = [
        Vertex {
            position: na::Vector3::new(-1.0, -1.0, 0.0),
            tex_coord: na::Vector2::new(0.0, 0.0),
        },
        Vertex {
            position: na::Vector3::new(1.0, -1.0, 0.0),
            tex_coord: na::Vector2::new(1.0, 0.0),
        },
        Vertex {
            position: na::Vector3::new(1.0, 1.0, 0.0),
            tex_coord: na::Vector2::new(1.0, 1.0),
        },
        Vertex {
            position: na::Vector3::new(-1.0, 1.0, 0.0),
            tex_coord: na::Vector2::new(0.0, 1.0),
        },
    ];

    const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

    fn new_quad(device: &Device, gfx_cmd_pool: vk::CommandPool) -> MemResult<VertexIndexBuffer> {
        let data = bytemuck::must_cast_slice(&Self::QUAD_VERTICES).iter()
            .chain(bytemuck::must_cast_slice(&Self::QUAD_INDICES).iter())
            .copied()
            .collect::<ArrayVec<u8, { std::mem::size_of::<[Vertex; 4]>() + std::mem::size_of::<[u16; 6]>() }>>();

        let mut buffer = Buffer::<GpuOnly>::new(
            utils::debug_only_name!("Quad vertex/index buffer"),
            &device,
            data.len() as u32,
            std::mem::size_of::<u8>(),
            vk::BufferUsageFlags::TRANSFER_DST |
                vk::BufferUsageFlags::VERTEX_BUFFER |
                vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;
        buffer.copy_via_staging_buffer(&device, &data, gfx_cmd_pool, QueueFamily::Graphics)?;

        Ok(VertexIndexBuffer::new(buffer,
                                  Self::QUAD_INDICES.len() as u32,
                                  vk::IndexType::UINT16,
                                  std::mem::size_of_val(&Self::QUAD_VERTICES) as vk::DeviceSize))
    }
}

