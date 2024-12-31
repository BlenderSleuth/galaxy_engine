// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::slice;
use std::sync::Arc;

use ash::vk;
use itertools::izip;

use crate::engine::GalaxyEngine;
use crate::maths::grid_size_for_count;
use crate::meshes::{Mesh, MeshError};
use crate::pipelines::{ComputeResourceType, Pipeline};
use crate::resource_paths::ResourcePath;
use crate::vertex_input::PositionTexCoordVertex;
use crate::volatile_buffer::{VolatileBuffer, VolatileBufferType};
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::{RecordingCmdBuf, RenderingState, TransientPrimaryCommandPool};
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::physical_device::PhysicalDevice;
use crate::vulkan::queue::queue_type::PrimaryQueue;

pub struct LoadingMeshManager {
    resource_path_map: HashMap<ResourcePath, u32>,
    meshes: Vec<Arc<Mesh>>,
    element_offset: u32,
}

impl LoadingMeshManager {
    pub(crate) fn new() -> Self {
        Self {
            resource_path_map: HashMap::new(),
            meshes: Vec::new(),
            element_offset: 0,
        }
    }

    pub fn get_or_load_mesh(
        &mut self,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        resource_path: &ResourcePath,
    ) -> Result<Arc<Mesh>, MeshError> {
        if let Some(mesh_index) = self.resource_path_map.get(resource_path) {
            Ok(Arc::clone(&self.meshes[*mesh_index as usize]))
        } else {
            let mesh_name = resource_path
                .path()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap_or("Unknown mesh");
            let mesh_index = self.meshes.len() as u32;
            let mesh = Arc::new(Mesh::new(
                mesh_name,
                engine,
                cmd_pool,
                resource_path,
                mesh_index,
                self.element_offset,
            )?);
            self.element_offset += mesh.num_elements();
            self.resource_path_map.insert(resource_path.clone(), mesh_index);
            self.meshes.push(Arc::clone(&mesh));
            Ok(mesh)
        }
    }

    pub(crate) fn finalise_loading(
        self,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<MeshManager> {
        MeshManager::new(self, engine, cmd_pool)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MegabufferPushConstants {
    group_count_x: u32,
    num_threads: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshBufferOffset {
    buffer_index: u32,
    offset: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct MeshElementDrawData {
    pub vertex_offset: i32,
    pub first_index: u32,
    pub index_count: u32,
}

// TODO: Use gpu-only buffers for offsets and buffer data.
pub struct MeshManager {
    _meshes: Vec<Arc<Mesh>>,
    element_draw_data: Vec<MeshElementDrawData>,
    vertex_megabuffer: Buffer<GpuOnly>,
    index_megabuffer: Buffer<GpuOnly>,
}

impl MeshManager {
    const NUM_MEGABUFFER_STORAGE_BUFFERS: usize = 6;
    const VERTEX_COPY_PIPELINE_ID: &'static str = "/engine/megabuffer/vertex_copy";
    const INDEX_COPY_PIPELINE_ID: &'static str = "/engine/megabuffer/index_copy";

    fn new(
        loading: LoadingMeshManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        let meshes = loading.meshes;
        // Loading manager should retain insertion order, so sorting again shouldn't be required.
        debug_assert!(meshes.is_sorted_by_key(|mesh| mesh.level_index));

        // Calculate total mesh stats.
        let num_meshes = meshes.len();
        let (num_vertices, num_indices) = meshes.iter().fold((0, 0), |acc, mesh| {
            (acc.0 + mesh.num_vertices(), acc.1 + mesh.num_indices())
        });

        // Calculate the draw data per mesh element.
        let element_draw_data = meshes
            .iter()
            .flat_map(|mesh| mesh.elements.iter())
            .scan((0, 0), |(vertex_offset, first_index), element| {
                let draw_data = MeshElementDrawData {
                    vertex_offset: *vertex_offset as i32,
                    first_index: *first_index,
                    index_count: element.index_count,
                };
                *vertex_offset += element.vertex_count;
                *first_index += element.index_count;
                Some(draw_data)
            })
            .collect();

        let (vertex_megabuffer, index_megabuffer) = {
            // Vertex offsets for the megabuffer construct shader (the buffer index and offset of each mesh element's indices).
            let mut vertex_offsets_buffer = VolatileBuffer::<MeshBufferOffset, 1>::new_array(
                "Level vertex offsets",
                num_vertices as usize,
                &engine.device,
                VolatileBufferType::Storage,
            )?;
            // Index offsets for the megabuffer construct shader (the buffer index and offset of each mesh element's indices).
            let mut index_offsets_buffer = VolatileBuffer::<MeshBufferOffset, 1>::new_array(
                "Level index offsets",
                num_indices as usize,
                &engine.device,
                VolatileBufferType::Storage,
            )?;

            // Set up ranges for copying vertices and indices into the megabuffer.
            let vertex_offsets_slice = vertex_offsets_buffer.get_mut_slice(0);
            let index_offsets_slice = index_offsets_buffer.get_mut_slice(0);
            meshes
                .iter()
                .enumerate()
                .fold((0, 0), |(vertex_offset, index_offset), (i, mesh)| {
                    let vertex_end = vertex_offset + mesh.num_vertices();
                    vertex_offsets_slice[vertex_offset as usize..vertex_end as usize].fill(MeshBufferOffset {
                        buffer_index: i as u32,
                        offset: vertex_offset,
                    });
                    let index_end = index_offset + mesh.num_indices();
                    index_offsets_slice[index_offset as usize..index_end as usize].fill(MeshBufferOffset {
                        buffer_index: i as u32,
                        offset: index_offset,
                    });
                    (vertex_end, index_end)
                });

            // Mesh buffer refs.
            let mut vertex_buffers = VolatileBuffer::<vk::DeviceAddress, 1>::new_array(
                "Level vertex buffer data",
                num_meshes,
                &engine.device,
                VolatileBufferType::Storage,
            )?;
            let mut index_buffers = VolatileBuffer::<vk::DeviceAddress, 1>::new_array(
                "Level index buffer data",
                num_meshes,
                &engine.device,
                VolatileBufferType::Storage,
            )?;
            for (vertex_buffer_addr, index_buffer_addr, mesh) in izip!(
                vertex_buffers.get_mut_slice(0).iter_mut(),
                index_buffers.get_mut_slice(0).iter_mut(),
                meshes.iter()
            ) {
                *vertex_buffer_addr = mesh.buffer().vertices_addr();
                *index_buffer_addr = mesh.buffer().indices_addr();
            }

            let vertices_size = (num_vertices as usize * size_of::<PositionTexCoordVertex>()) as vk::DeviceSize;
            let vertex_megabuffer = Buffer::new(
                "Level vertex megabuffer",
                &engine.device,
                vertices_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            let indices_size = (num_indices as usize * size_of::<u32>()) as vk::DeviceSize;
            let index_megabuffer = Buffer::new(
                "Level index megabuffer",
                &engine.device,
                indices_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            // Set up compute descriptor set.
            let mut megabuffer_descriptor_pool = DescriptorPool::<1>::new(
                &engine.device,
                &[vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(Self::NUM_MEGABUFFER_STORAGE_BUFFERS as u32)],
            )?;
            let megabuffer_descriptor_set = {
                // Set up megabuffer descriptor set.
                let megabuffer_descriptor_set_layout = engine
                    .pipeline_manager
                    .get_compute_descriptor_set_layout(
                        &[ComputeResourceType::StorageBuffer; Self::NUM_MEGABUFFER_STORAGE_BUFFERS],
                    )
                    .unwrap();
                megabuffer_descriptor_pool
                    .allocate_descriptor_sets(&engine.device, &[megabuffer_descriptor_set_layout])?;
                megabuffer_descriptor_pool.get(0)
            };
            {
                let vertex_offsets_buffer_info = vertex_offsets_buffer.descriptor_buffer_info(0);
                let vertex_buffers_buffer_info = vertex_buffers.descriptor_buffer_info(0);
                let vertex_megabuffer_info = vertex_megabuffer.descriptor_buffer_info();

                let index_offsets_buffer_info = index_offsets_buffer.descriptor_buffer_info(0);
                let index_buffers_info = index_buffers.descriptor_buffer_info(0);
                let index_megabuffer_info = index_megabuffer.descriptor_buffer_info();

                let descriptor_writes: [vk::WriteDescriptorSet; Self::NUM_MEGABUFFER_STORAGE_BUFFERS] = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&vertex_offsets_buffer_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&vertex_buffers_buffer_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&vertex_megabuffer_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&index_offsets_buffer_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&index_buffers_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(megabuffer_descriptor_set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&index_megabuffer_info)),
                ];
                unsafe { engine.device.loader().update_descriptor_sets(&descriptor_writes, &[]) };
            }

            // Run copy compute shaders.
            {
                let vertex_copy_pipeline = engine
                    .pipeline_manager
                    .get_compute_pipeline(Self::VERTEX_COPY_PIPELINE_ID)
                    .expect("Vertex copy pipeline not found");
                let index_copy_pipeline = engine
                    .pipeline_manager
                    .get_compute_pipeline(Self::INDEX_COPY_PIPELINE_ID)
                    .expect("Index copy pipeline not found");
                let vertex_total_group_count = num_vertices.div_ceil(vertex_copy_pipeline.total_threads_per_group());
                let (vertex_group_count_x, vertex_group_count_y) = grid_size_for_count(vertex_total_group_count, 5);
                assert!(vertex_group_count_x * vertex_group_count_y >= vertex_total_group_count);
                assert!(vertex_group_count_x <= PhysicalDevice::MAX_DISPATCH_GROUPS_PER_DIMENSION);
                assert!(vertex_group_count_y <= PhysicalDevice::MAX_DISPATCH_GROUPS_PER_DIMENSION);

                let index_total_group_count = num_indices.div_ceil(index_copy_pipeline.total_threads_per_group());
                let (index_group_count_x, index_group_count_y) = grid_size_for_count(index_total_group_count, 5);
                assert!(index_group_count_x * index_group_count_y >= index_total_group_count);
                assert!(index_group_count_x <= PhysicalDevice::MAX_DISPATCH_GROUPS_PER_DIMENSION);
                assert!(index_group_count_y <= PhysicalDevice::MAX_DISPATCH_GROUPS_PER_DIMENSION);

                let mut cmd_buf = cmd_pool.allocate_transient_cmd_buffer()?;
                cmd_buf.bind_descriptor_sets(
                    vk::PipelineBindPoint::COMPUTE,
                    vertex_copy_pipeline.layout(),
                    0,
                    &[megabuffer_descriptor_set],
                    &[],
                );

                cmd_buf.bind_compute_pipeline(vertex_copy_pipeline);
                cmd_buf.push_constants(
                    vertex_copy_pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&MegabufferPushConstants {
                        group_count_x: vertex_group_count_x,
                        num_threads: num_vertices,
                    }),
                );
                //let copy_barrier = vk::BufferMemoryBarrier2::default()
                //    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                //    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                //    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                //    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                //    .buffer(index_buffers.handle_dep())
                //    .size(vk::WHOLE_SIZE)
                //    .offset(0);
                //let dependency_info = vk::DependencyInfo::default().memory_barriers(slice::from_ref(&copy_barrier));
                //cmd_buf.pipeline_barrier2(&engine.device, &dependency_info);
                cmd_buf.dispatch(vertex_group_count_x, vertex_group_count_y, 1);

                cmd_buf.bind_compute_pipeline(index_copy_pipeline);
                cmd_buf.push_constants(
                    vertex_copy_pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&MegabufferPushConstants {
                        group_count_x: index_group_count_x,
                        num_threads: num_indices,
                    }),
                );
                cmd_buf.dispatch(index_group_count_x, index_group_count_y, 1);

                cmd_buf.end_submit_wait_and_free()?;
            }

            (vertex_megabuffer, index_megabuffer)
        };

        Ok(Self {
            _meshes: meshes,
            element_draw_data,
            vertex_megabuffer,
            index_megabuffer,
        })
    }

    pub(crate) fn bind_megabuffer(&self, cmd_buf: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buf.bind_vertex_buffer(&self.vertex_megabuffer, 0);
        cmd_buf.bind_index_buffer(&self.index_megabuffer, 0, vk::IndexType::UINT32);
    }

    pub(crate) fn get_element_draw_data_for_mesh(&self, mesh: &Mesh) -> &[MeshElementDrawData] {
        &self.element_draw_data[mesh.level_element_range()]
    }
}
