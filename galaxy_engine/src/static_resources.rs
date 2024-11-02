// Copyright (c) 2024 Ben Sutherland.

use std::f32::consts::FRAC_1_SQRT_2;
use std::ops::Deref;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::mesh::{MeshBuffer, Vertex};
use crate::prelude::*;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;

const FRAC_1_SQRT_8: f32 = FRAC_1_SQRT_2 * 0.5;

pub(crate) type StaticResourcesLock = RwLock<Option<StaticResources>>;
pub(crate) type StaticResourcesRef = &'static StaticResourcesLock;
pub(crate) struct StaticResourcesGuard {
    resources: StaticResourcesRef,
}

impl StaticResourcesGuard {
    pub fn new(resources: StaticResourcesRef) -> Self {
        Self { resources }
    }
}

impl Drop for StaticResourcesGuard {
    fn drop(&mut self) {
        *self.resources.write() = None;
    }
}

pub struct StaticResources {
    quad_buffer: Arc<MeshBuffer>,
    octagon_buffer: Arc<MeshBuffer>,
}

impl StaticResources {
    /// Initialisation of static resources.
    pub(crate) fn new(device: &Device, cmd_pool: &mut TransientPrimaryCommandPool) -> MemResult<Self> {
        let mut cmd_buffer = cmd_pool.allocate_transient_cmd_buffer()?;
        let result = Self {
            quad_buffer: Arc::new(MeshBuffer::new_from_vertices_and_indices(
                "Quad",
                &Self::QUAD_VERTICES,
                &Self::QUAD_INDICES,
                device,
                &mut cmd_buffer,
            )?),
            octagon_buffer: Arc::new(MeshBuffer::new_from_vertices_and_indices(
                "Octagon",
                &Self::OCTAGON_VERTICES,
                &Self::OCTAGON_INDICES,
                device,
                &mut cmd_buffer,
            )?),
        };
        cmd_buffer.end_submit_wait_and_free()?;

        Ok(result)
    }

    /// Quad vertex/index buffer.
    pub fn get_quad(&self) -> &MeshBuffer {
        self.quad_buffer.deref()
    }
    pub fn get_quad_cloned(&self) -> Arc<MeshBuffer> {
        Arc::clone(&self.quad_buffer)
    }

    const QUAD_VERTICES: [Vertex; 4] = [
        Vertex {
            position: Vec3::new(-1.0, -1.0, 0.0),
            tex_coord: Vec2::new(0.0, 0.0),
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 0.0),
            tex_coord: Vec2::new(1.0, 0.0),
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, 0.0),
            tex_coord: Vec2::new(1.0, 1.0),
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, 0.0),
            tex_coord: Vec2::new(0.0, 1.0),
        },
    ];

    const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

    /// Octagon vertex/index buffer.
    pub fn get_octagon(&self) -> &MeshBuffer {
        self.octagon_buffer.deref()
    }
    pub fn get_octagon_cloned(&self) -> Arc<MeshBuffer> {
        Arc::clone(&self.octagon_buffer)
    }

    const OCTAGON_VERTICES: [Vertex; 8] = [
        Vertex {
            position: Vec3::new(1.0, 0.0, 0.0),
            tex_coord: Vec2::new(1.0, 0.5),
        },
        Vertex {
            position: Vec3::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
            tex_coord: Vec2::new(0.5 + FRAC_1_SQRT_8, 0.5 + FRAC_1_SQRT_8),
        },
        Vertex {
            position: Vec3::new(0.0, 1.0, 0.0),
            tex_coord: Vec2::new(0.5, 1.0),
        },
        Vertex {
            position: Vec3::new(-FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
            tex_coord: Vec2::new(0.5 - FRAC_1_SQRT_8, 0.5 + FRAC_1_SQRT_8),
        },
        Vertex {
            position: Vec3::new(-1.0, 0.0, 0.0),
            tex_coord: Vec2::new(0.0, 0.5),
        },
        Vertex {
            position: Vec3::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
            tex_coord: Vec2::new(0.5 - FRAC_1_SQRT_8, 0.5 - FRAC_1_SQRT_8),
        },
        Vertex {
            position: Vec3::new(0.0, -1.0, 0.0),
            tex_coord: Vec2::new(0.5, 0.0),
        },
        Vertex {
            position: Vec3::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
            tex_coord: Vec2::new(0.5 + FRAC_1_SQRT_8, 0.5 - FRAC_1_SQRT_8),
        },
    ];

    const OCTAGON_INDICES: [u16; 18] = [0, 1, 2, 0, 2, 7, 2, 3, 7, 3, 6, 7, 3, 4, 6, 4, 5, 6];
}
