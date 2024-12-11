// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::GalaxyEngine;
use crate::meshes::{Mesh, MeshError};
use crate::resource_paths::ResourcePath;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;

pub struct MeshManager {
    meshes: HashMap<ResourcePath, Arc<Mesh>>,
}

impl MeshManager {
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
