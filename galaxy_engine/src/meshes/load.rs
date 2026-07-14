// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use bevy_mikktspace::Geometry;
use meshopt::VertexDataAdapter;
use obj::raw::object::Polygon;
use ultraviolet::{Vec2, Vec3};

use crate::meshes::MeshElementOffset;
use crate::prelude::*;
use crate::vertex_input::MeshVertex;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawMeshVertex {
    position: Vec3,
    normal: Vec3,
    tex_coord: Vec2,
    tangent: Vec3,
    handedness: f32,
}

pub struct RawMeshVertices {
    pub vertices: Vec<RawMeshVertex>,
    pub indices: Vec<u32>,
}

impl RawMeshVertices {
    fn index_for_face_vertex(&self, face: usize, vert: usize) -> usize {
        self.indices[face * 3 + vert] as usize
    }

    fn get_vertex(&self, face: usize, vert: usize) -> &RawMeshVertex {
        &self.vertices[self.index_for_face_vertex(face, vert)]
    }

    fn get_vertex_mut(&mut self, face: usize, vert: usize) -> &mut RawMeshVertex {
        let index = self.index_for_face_vertex(face, vert);
        &mut self.vertices[index]
    }
}

impl Geometry for RawMeshVertices {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let vertex = self.get_vertex(face, vert);
        [vertex.position.x, vertex.position.y, vertex.position.z]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let vertex = self.get_vertex(face, vert);
        [vertex.normal.x, vertex.normal.y, vertex.normal.z]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let vertex = self.get_vertex(face, vert);
        [vertex.tex_coord.x, vertex.tex_coord.y]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let vertex = self.get_vertex_mut(face, vert);
        vertex.tangent = Vec3::new(tangent[0], tangent[1], tangent[2]).normalized();
        vertex.handedness = tangent[3];
    }
}

pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub elements: Vec<MeshElementOffset>,
}

impl MeshData {
    fn process_vertices(raw: &[RawMeshVertex], indices: &[u32]) -> Vec<MeshVertex> {
        //let start = std::time::Instant::now();
        // Construct adjacency information.
        let mut adjacency = vec![smallvec::SmallVec::<[u32; 8]>::new(); indices.len()]; // Vertices usually aren't connected to more than 8 neighbours.
        for tri in indices.chunks_exact(3) {
            adjacency[tri[0] as usize].push(tri[1]);
            adjacency[tri[0] as usize].push(tri[2]);
            adjacency[tri[1] as usize].push(tri[0]);
            adjacency[tri[1] as usize].push(tri[2]);
            adjacency[tri[2] as usize].push(tri[0]);
            adjacency[tri[2] as usize].push(tri[1]);
        }
        //log::info!("Constructed adjacency in {:?}", start.elapsed());

        // Calculate quaternion tangent frames.
        //let start = std::time::Instant::now();
        let mut quats: Vec<_> = raw
            .iter()
            .map(|v| {
                // Calculate right-handed TBN matrix.
                let normal = v.normal; // Normal is normalised on load.
                let tangent = v.tangent; // Tangent is normalised when calculated.
                let bitangent = normal.cross(tangent).normalized(); // Should panic if something went wrong.
                let tangent = bitangent.cross(normal); // Ensure orthogonality.

                let tbn_mat = Mat3::new(tangent, bitangent, normal);
                tbn_mat.into_rotor3().normalized()
            })
            .collect();
        //log::info!("Calculated quaternions in {:?}", start.elapsed());

        // Align quaternions so they interpolate correctly across triangles.
        //let start = std::time::Instant::now();
        let mut traversed_list = vec![false; raw.len()];
        // For each untraversed vertex.
        for v_idx in 0..raw.len() {
            if traversed_list[v_idx] {
                continue;
            }
            traversed_list[v_idx] = true;
            let mut stack = vec![v_idx as u32];
            while let Some(v_idx) = stack.pop() {
                for &adj in adjacency[v_idx as usize].iter() {
                    if !traversed_list[adj as usize] {
                        traversed_list[adj as usize] = true;
                        if quats[v_idx as usize].dot(quats[adj as usize]) < 0. {
                            quats[adj as usize] *= -1.;
                        }

                        stack.push(adj);
                    }
                }
            }
        }
        //log::info!("Aligned quaternions in {:?}", start.elapsed());

        // Quantise and pack quaternions.
        //let start = std::time::Instant::now();
        let result = raw
            .iter()
            .zip(quats)
            .map(|(v, q)| {
                let quaternion = rotor_to_shader_quat(q);
                // Quantise quaternion.
                let mut qtangent = [0u8; 4];
                qtangent
                    .iter_mut()
                    .zip(quaternion.iter())
                    .take(3)
                    .for_each(|(snorm, component)| {
                        *snorm = to_unorm(component * 0.5 + 0.5);
                    });

                // Pack the w component (see GPU Pro 5, p.361, "Quaternions Revisited").
                // In the source for that article the high bit encodes whether the handedness is inverted (0 = inverted, 1 = standard).
                let high_bit: u8 = if v.handedness > 0. { 0x80 } else { 0 };
                let quantised_w = to_unorm(quaternion[3] * 0.5 + 0.5) >> 1;
                debug_assert!(quantised_w <= 127);
                qtangent[3] = high_bit | quantised_w;

                MeshVertex {
                    position: v.position,
                    qtangent,
                    tex_coord: v.tex_coord,
                }
            })
            .collect();
        //log::info!("Quantised quaternions in {:?}", start.elapsed());

        result
    }

    pub fn load_obj(obj_path: &Path) -> Result<Self, obj::ObjError> {
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
        let mut map = |pi: usize, ni: usize, ti: usize| -> u32 {
            // Look up cache
            match cache.entry((pi, ni, ti)) {
                // Cache miss -> make new, store it on cache.
                Entry::Vacant(entry) => {
                    let p = positions[pi];
                    let n = normals[ni];
                    let t = tex_coords[ti];
                    let vertex = RawMeshVertex {
                        position: Vec3::new(p.0, p.1, p.2),
                        tex_coord: Vec2::new(t.0, 1. - t.1),
                        //tex_coord: Vec2::new(t.0, t.1),
                        normal: Vec3::new(n.0, n.1, n.2).normalized(),
                        tangent: Vec3::zero(),
                        handedness: 1.,
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
                        Polygon::PTN(vec) if vec.len() == 3 => {
                            let triangle = (
                                core::array::from_fn::<_, 3, _>(|i| {
                                    let (pi, ti, ni) = vec[i];
                                    map(pi, ni, ti)
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
                std::mem::size_of::<RawMeshVertex>(),
                std::mem::offset_of!(RawMeshVertex, position),
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

        let start = std::time::Instant::now();
        let mut raw_mesh = RawMeshVertices { vertices, indices: ib };
        assert!(bevy_mikktspace::generate_tangents(&mut raw_mesh));
        log::info!("Generated tangents in {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let vertices = Self::process_vertices(&raw_mesh.vertices, &raw_mesh.indices);
        log::info!("Processed vertices in {:?}", start.elapsed());

        Ok(Self {
            vertices,
            indices: raw_mesh.indices,
            elements,
        })
    }
}
