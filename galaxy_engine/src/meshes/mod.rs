// Copyright (c) 2024 Ben Sutherland.

pub mod mesh_manager;

use std::alloc::Layout;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::num::TryFromIntError;
use std::path::Path;

use ash::vk;
use meshopt::VertexDataAdapter;
use obj::raw::object::Polygon;

use crate::engine::GalaxyEngine;
use crate::prelude::*;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::vertex_input::PositionTexCoordVertex;
use crate::vulkan::buffer::{Buffer, CpuToGpu, GpuOnly};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState, TransientPrimaryCommandPool};
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
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        // Ensure proper alignment.
        let indices_offset = Self::pad_vertices_size::<I>(size_of_val(vertices));
        let buffer_size = (indices_offset + size_of_val(indices)) as vk::DeviceSize;

        let mut buffer = Buffer::<GpuOnly>::new(
            debug::debug_only_name!("{name} meshes buffer"),
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST
                //| vk::BufferUsageFlags::VERTEX_BUFFER
                //| vk::BufferUsageFlags::INDEX_BUFFER 
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            None,
        )?;

        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            debug::debug_only_name!("{name} meshes staging buffer"),
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            None,
        )?;
        staging_buffer.copy_slice_into_buffer(vertices, 0)?;
        staging_buffer.copy_slice_into_buffer(indices, indices_offset)?;

        let mut cmd_buffer = cmd_pool.allocate_transient_cmd_buffer()?;
        staging_buffer.copy_to_buffer(&mut cmd_buffer, &mut buffer, staging_buffer.size());
        cmd_buffer.end_submit_wait_and_free()?;

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

    #[deprecated]
    pub fn bind(&self, cmd_buffer: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buffer.bind_vertex_buffer(&self.buffer, 0);
        cmd_buffer.bind_index_buffer(&self.buffer, self.indices_offset, self.index_type);
    }

    //pub fn draw(&self, cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>, first_index: u32, vertex_offset: i32) {
    //    cmd_buf.draw_indexed(self.num_indices(), 1, first_index, vertex_offset, 0);
    //}
}

struct LoadedObj {
    vertices: Vec<PositionTexCoordVertex>,
    indices: Vec<u32>,
    num_elements: u32,
}

fn load_obj(obj_path: &Path) -> Result<LoadedObj, obj::ObjError> {
    let mtl_path = obj_path.with_extension("mtl");

    // Load model.
    let start = std::time::Instant::now();
    let raw_obj = obj::raw::parse_obj(BufReader::new(File::open(obj_path)?))?;

    // Get ordered mesh elements (based on material).
    let mut element_index = 0;
    let mut element_orders = HashMap::new();
    let mtl_str = std::fs::read_to_string(mtl_path);
    if let Ok(mtl_str) = mtl_str.as_ref() {
        for line in mtl_str.lines() {
            let mut parts = line.split_whitespace();
            if let Some("newmtl") = parts.next() {
                let name = parts.next().expect("Material name not found");
                element_orders.insert(name, element_index);
                element_index += 1;
            }
        }
    }
    // Require at least one element.
    let num_elements = element_index.max(1);

    // Index vertices.
    let polygons = &raw_obj.polygons;
    let positions = &raw_obj.positions;
    let normals = &raw_obj.normals;
    let tex_coords = &raw_obj.tex_coords;
    let mut vb = Vec::with_capacity(polygons.len() * 3);
    let mut ib = Vec::with_capacity(polygons.len() * 3);

    // Indexing code from obj crate.
    let mut cache = HashMap::new();
    let mut can_use_16_bit = true;
    let mut map = |pi: usize, ni: usize, ti: usize, element_index: u32| -> Result<(), TryFromIntError> {
        // Look up cache
        let index = match cache.entry((pi, element_index, ti)) {
            // Cache miss -> make new, store it on cache
            Entry::Vacant(entry) => {
                let p = positions[pi];
                let _n = normals[ni];
                let t = tex_coords[ti];
                let vertex = PositionTexCoordVertex {
                    position: Vec3::new(p.0, p.1, p.2),
                    element_index,
                    tex_coord: Vec2::new(t.0, 1. - t.1),
                };

                let index = u32::try_from(vb.len())?;
                if u16::try_from(index).is_err() {
                    can_use_16_bit = false;
                }
                vb.push(vertex);
                entry.insert(index);
                index
            }
            // Cache hit -> use it
            Entry::Occupied(entry) => *entry.get(),
        };
        ib.push(index);
        Ok(())
    };
    raw_obj.meshes.iter().for_each(|(mat, group)| {
        let element_index = element_orders.get(mat.as_str()).copied().unwrap_or(0);
        group.polygons.iter().for_each(|range| {
            polygons[range.start..range.end]
                .iter()
                .for_each(|polygon| match polygon {
                    Polygon::P(_) => {
                        panic!("Tried to extract normal and texture data which are not contained in the model")
                    }
                    Polygon::PT(_) => panic!("Tried to extract normal data which are not contained in the model"),
                    Polygon::PN(_) => panic!("Tried to extract texture data which are not contained in the model"),
                    Polygon::PTN(ref vec) if vec.len() == 3 => {
                        for &(pi, ti, ni) in vec {
                            map(pi, ni, ti, element_index).unwrap()
                        }
                    }
                    _ => panic!("Model should be triangulated first to be loaded properly"),
                })
        })
    });
    log::info!("Loaded mesh in {:?}", start.elapsed());

    // Optimize model.
    let start = std::time::Instant::now();
    let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vb, Some(&ib));
    let mut vertices = meshopt::remap_vertex_buffer(&vb, vertex_count, &vert_remap);
    let mut indices = meshopt::remap_index_buffer(Some(&ib), vertex_count, &vert_remap);
    meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);
    let vertex_data_adapter = VertexDataAdapter::new(
        bytemuck::must_cast_slice(&vertices),
        std::mem::size_of::<PositionTexCoordVertex>(),
        std::mem::offset_of!(PositionTexCoordVertex, position),
    )
    .unwrap();
    meshopt::optimize_overdraw_in_place(&mut indices, &vertex_data_adapter, 1.05);
    meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);
    log::info!("Optimized mesh in {:?}", start.elapsed());

    Ok(LoadedObj {
        vertices,
        indices,
        num_elements,
    })
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
    mesh_buffer: MeshBuffer<PositionTexCoordVertex>,
    num_elements: u32,
}

impl Mesh {
    pub fn new(
        name: &str,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        mesh_path: &ResourcePath,
    ) -> Result<Self, MeshError> {
        let obj_path = mesh_path.full_path::<resource_type::Mesh>(engine);
        let loaded_obj = load_obj(&obj_path)?;
        let mesh_buffer = MeshBuffer::new_from_vertices_and_indices(
            name,
            &loaded_obj.vertices,
            &loaded_obj.indices,
            &engine.device,
            cmd_pool,
        )?;
        log::info!(
            "Loaded mesh: {} has {} vertices and {} indices.",
            name,
            mesh_buffer.num_vertices(),
            mesh_buffer.num_indices()
        );
        Ok(Self {
            mesh_buffer,
            num_elements: loaded_obj.num_elements,
        })
    }

    pub fn num_vertices(&self) -> u32 {
        self.mesh_buffer.num_vertices()
    }

    pub fn num_indices(&self) -> u32 {
        self.mesh_buffer.num_indices()
    }

    pub fn num_elements(&self) -> u32 {
        self.num_elements
    }

    pub fn buffer(&self) -> &MeshBuffer<PositionTexCoordVertex> {
        &self.mesh_buffer
    }
}
