// Copyright (c) 2024 Ben Sutherland.

pub mod mesh_manager;

use std::alloc::Layout;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ash::vk;
use meshopt::VertexDataAdapter;
use obj::raw::object::Polygon;

use crate::engine::GalaxyEngine;
use crate::prelude::*;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::vertex_input::PositionTexCoordVertex;
use crate::vulkan::buffer::{Buffer, GpuOnly, Staging};
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
        )?;

        let mut staging_buffer = Buffer::<Staging>::new(
            debug::debug_only_name!("{name} meshes staging buffer"),
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
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

struct MeshElementOffset {
    _vertex_offset: u32,
    vertex_count: u32,
    _index_offset: u32,
    index_count: u32,
}

struct LoadedObj {
    vertices: Vec<PositionTexCoordVertex>,
    indices: Vec<u32>,
    elements: Vec<MeshElementOffset>,
}

fn load_obj(obj_path: &Path) -> Result<LoadedObj, obj::ObjError> {
    let mtl_path = obj_path.with_extension("mtl");

    // Load model.
    let load_start = std::time::Instant::now();
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
    let mut map = |pi: usize, ni: usize, ti: usize, element_index: u32| -> u32 {
        // Look up cache
        match cache.entry((pi, ti, element_index)) {
            // Cache miss -> make new, store it on cache.
            Entry::Vacant(entry) => {
                let p = positions[pi];
                let _n = normals[ni];
                let t = tex_coords[ti];
                let vertex = PositionTexCoordVertex {
                    position: Vec3::new(p.0, p.1, p.2),
                    element_index,
                    tex_coord: Vec2::new(t.0, 1. - t.1),
                };

                let index = u32::try_from(vb.len())
                    .unwrap_or_else(|_| panic!("Mesh {obj_path:?} contains over u32::MAX vertices."));
                if u16::try_from(index).is_err() {
                    can_use_16_bit = false;
                }
                vb.push(vertex);
                entry.insert(index);
                index
            }
            // Cache hit -> use it.
            Entry::Occupied(entry) => *entry.get(),
        }
    };
    raw_obj.meshes.iter().for_each(|(mat, group)| {
        let element_index = element_orders.get(mat.as_str()).copied().unwrap_or(0) as u32;

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
                        let triangle = (
                            core::array::from_fn::<_, 3, _>(|i| {
                                let (pi, ti, ni) = vec[i];
                                map(pi, ni, ti, 0)
                            }),
                            element_index,
                        );
                        ib.push(triangle);
                    }
                    _ => panic!("Model should be triangulated first to be loaded properly"),
                })
        });
    });

    // Sort triangles by element index.
    ib.sort_by_key(|i| i.1);

    // Calculate the number and offset of indices for each element.
    let element_index_ranges = (0..num_elements)
        .scan(0, |start_index, _| {
            let element_index = ib[*start_index].1;
            let end_index = ib
                .iter()
                .skip(*start_index)
                .position(|&i| i.1 != element_index)
                .map(|p| *start_index + p)
                .unwrap_or(ib.len());
            let index_range = (*start_index * 3)..(end_index * 3);
            *start_index = end_index;
            Some(index_range)
        })
        .collect::<Vec<_>>();

    let mut ib: Vec<u32> = ib.into_iter().flat_map(|(tri, _)| tri).collect();

    log::info!("Loaded mesh in {:?}", load_start.elapsed());

    let start = std::time::Instant::now();
    // Optimize each element.
    let mut vertices = Vec::with_capacity(vb.len());
    let mut elements = Vec::with_capacity(num_elements as usize);
    for element_index_range in element_index_ranges {
        let old_element_indices = &mut ib[element_index_range.clone()];

        let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vb, Some(old_element_indices));
        let mut element_vertices = meshopt::remap_vertex_buffer(&vb, vertex_count, &vert_remap);
        let mut element_indices = meshopt::remap_index_buffer(Some(old_element_indices), vertex_count, &vert_remap);
        assert_eq!(element_indices.len(), old_element_indices.len()); // mesh-opt shouldn't find any duplicates.

        meshopt::optimize_vertex_cache_in_place(&mut element_indices, vertex_count);
        let vertex_data_adapter = VertexDataAdapter::new(
            bytemuck::must_cast_slice(&element_vertices),
            std::mem::size_of::<PositionTexCoordVertex>(),
            std::mem::offset_of!(PositionTexCoordVertex, position),
        )
        .unwrap();
        meshopt::optimize_overdraw_in_place(&mut element_indices, &vertex_data_adapter, 1.05);
        meshopt::optimize_vertex_fetch_in_place(&mut element_indices, &mut element_vertices);

        // Copy over to overall mesh buffer.
        let vertex_offset = vertices.len() as u32;
        let index_offset = element_index_range.start as u32;
        vertices.extend_from_slice(&element_vertices);
        old_element_indices.copy_from_slice(&element_indices);
        elements.push(MeshElementOffset {
            _vertex_offset: vertex_offset,
            vertex_count: element_vertices.len() as u32,
            _index_offset: index_offset,
            index_count: element_indices.len() as u32,
        });
    }

    log::info!("Optimized mesh in {:?}", start.elapsed());

    Ok(LoadedObj {
        vertices,
        indices: ib,
        elements,
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
    buffer: MeshBuffer<PositionTexCoordVertex>,
    level_index: u32,          // Index of the mesh in the level.
    level_element_offset: u32, // Offset of the first element in the mesh in the level.
    elements: Vec<MeshElementOffset>,
}

impl Mesh {
    pub fn new(
        name: &str,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        mesh_path: &ResourcePath,
        level_index: u32,
        level_element_offset: u32,
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

    pub fn buffer(&self) -> &MeshBuffer<PositionTexCoordVertex> {
        &self.buffer
    }
}
