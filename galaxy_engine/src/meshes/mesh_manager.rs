// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::engine::GalaxyEngine;
use crate::meshes::{Mesh, MeshError};
use crate::resource_paths::ResourcePath;
use crate::vertex_input::PositionTexCoordVertex;
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;

pub struct LoadingMeshManager {
    meshes: HashMap<ResourcePath, Arc<Mesh>>,
}

impl LoadingMeshManager {
    pub(crate) fn new() -> Self {
        Self { meshes: HashMap::new() }
    }

    pub fn get_or_load_mesh(
        &mut self,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        resource_path: &ResourcePath,
    ) -> Result<Arc<Mesh>, MeshError> {
        if let Some(mesh) = self.meshes.get(resource_path) {
            Ok(Arc::clone(mesh))
        } else {
            let mesh_name = resource_path
                .path()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap_or("Unknown mesh");
            let mesh = Arc::new(Mesh::new(mesh_name, engine, cmd_pool, resource_path)?);
            self.meshes.insert(resource_path.clone(), Arc::clone(&mesh));
            Ok(mesh)
        }
    }
}

pub struct MeshManager {
    meshes: HashMap<ResourcePath, Arc<Mesh>>,
    vertices_megabuffer: Buffer<GpuOnly>,
    indices_megabuffer: Buffer<GpuOnly>,
    //element_offset_megabuffer: Buffer<GpuOnly>,
}

impl MeshManager {
    pub(crate) fn new(loading: LoadingMeshManager, device: &Device) -> MemResult<Self> {
        let (num_vertices, num_indices) = loading.meshes.values().fold((0, 0), |acc, mesh| {
            (acc.0 + mesh.num_vertices(), acc.1 + mesh.num_indices())
        });

        let vertices_size = (num_vertices as usize * size_of::<PositionTexCoordVertex>()) as vk::DeviceSize;
        let vertices_megabuffer = Buffer::new(
            "Level vertices megabuffer",
            device,
            vertices_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            None,
        )?;

        let indices_size = (num_indices as usize * size_of::<u32>()) as vk::DeviceSize;
        let indices_megabuffer = Buffer::new(
            "Level indices megabuffer",
            device,
            indices_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            None,
        )?;

        Ok(Self {
            meshes: loading.meshes,
            vertices_megabuffer,
            indices_megabuffer,
        })
    }

    //pub fn fill_mega_buffer(&self, engine: &GalaxyEngine) -> VkResult<()> {
    //    Ok(())
    //}
}
