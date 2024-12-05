// Copyright (c) 2024 Ben Sutherland.

use crate::meshes::Mesh;

pub struct MeshManager {
    meshes: Vec<Mesh>,
}

impl MeshManager {
    pub fn new() -> Self {
        Self { meshes: Vec::new() }
    }
}
