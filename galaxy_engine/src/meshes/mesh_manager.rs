// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::slice;
use std::sync::Arc;

use ash::vk;
use itertools::{izip, Itertools};

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

#[derive(Default)]
pub struct MeshDrawOffset {
    pub vertex_offset: i32,
    pub index_offset: u32,
}

pub struct MeshManager {
    meshes: Vec<Arc<Mesh>>,
    mesh_draw_offsets: Vec<MeshDrawOffset>,
    // Buffers:
    _vertex_offsets_buffer: VolatileBuffer<MeshBufferOffset, 1>,
    _vertex_buffers: VolatileBuffer<vk::DeviceAddress, 1>,
    vertex_megabuffer: Buffer<GpuOnly>,
    _index_offsets_buffer: VolatileBuffer<MeshBufferOffset, 1>,
    _index_buffers: VolatileBuffer<vk::DeviceAddress, 1>,
    index_megabuffer: Buffer<GpuOnly>,
    // Descriptors:
    _megabuffer_descriptor_pool: DescriptorPool<1>,
}

impl MeshManager {
    const NUM_MEGABUFFER_STORAGE_BUFFERS: usize = 6;
    const VERTEX_COPY_PIPELINE_ID: &'static str = "/engine/megabuffer/vertex_copy";
    const INDEX_COPY_PIPELINE_ID: &'static str = "/engine/megabuffer/index_copy";

    pub(crate) fn new(
        loading: LoadingMeshManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        // Set up megabuffer descriptor set.
        let megabuffer_descriptor_set_layout = engine
            .pipeline_manager
            .get_compute_descriptor_set_layout(
                &[ComputeResourceType::StorageBuffer; Self::NUM_MEGABUFFER_STORAGE_BUFFERS],
            )
            .unwrap();
        let mut megabuffer_descriptor_pool = DescriptorPool::new(
            &engine.device,
            &[vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(Self::NUM_MEGABUFFER_STORAGE_BUFFERS as u32)],
        )?;
        megabuffer_descriptor_pool.allocate_descriptor_sets(&engine.device, &[megabuffer_descriptor_set_layout])?;
        let megabuffer_descriptor_set = megabuffer_descriptor_pool.get(0);

        // Set up megabuffer storage buffers.
        let meshes = loading.meshes.into_values().collect::<Vec<_>>();
        let num_meshes = meshes.len();

        let mut mesh_draw_offsets = Vec::<MeshDrawOffset>::new();
        mesh_draw_offsets.resize_with(num_meshes, Default::default);

        let (num_vertices, num_indices) = meshes.iter().fold((0, 0), |acc, mesh| {
            (acc.0 + mesh.num_vertices(), acc.1 + mesh.num_indices())
        });
        // Vertex megabuffer. TODO: Use gpu-only buffers for offsets and buffer data.
        let vertices_size = (num_vertices as usize * size_of::<PositionTexCoordVertex>()) as vk::DeviceSize;
        log::info!("Total vertices: {}, size: {}", num_vertices, vertices_size);

        // Offsets.
        let mut vertex_offsets_buffer = VolatileBuffer::<MeshBufferOffset, 1>::new_array(
            "Level vertex offsets",
            num_vertices as usize,
            &engine.device,
            VolatileBufferType::Storage,
        )?;
        let vertex_offsets = meshes.iter().scan(0, |offset, mesh| {
            let result = *offset;
            *offset += mesh.num_vertices();
            Some(result)
        });
        // Iterate over the ranges of vertices for each mesh.
        let vertex_offsets_slice = vertex_offsets_buffer.get_mut_slice(0);
        for (i, (draw_offset, (start, end))) in izip!(
            mesh_draw_offsets.iter_mut(),
            vertex_offsets.chain(std::iter::once(num_vertices)).tuple_windows(),
        )
        .enumerate()
        {
            draw_offset.vertex_offset = start as i32;
            let vertex_offset = MeshBufferOffset {
                buffer_index: i as u32,
                offset: start,
            };
            vertex_offsets_slice[start as usize..end as usize].fill(vertex_offset);
        }
        let vertex_offsets_buffer_info = vertex_offsets_buffer.descriptor_buffer_info(0);

        // Vertex buffers data.
        let mut vertex_buffers = VolatileBuffer::<vk::DeviceAddress, 1>::new_array(
            "Level vertex buffer data",
            num_meshes,
            &engine.device,
            VolatileBufferType::Storage,
        )?;
        for (vertex_buffer_addr, mesh) in vertex_buffers.get_mut_slice(0).iter_mut().zip(meshes.iter()) {
            *vertex_buffer_addr = mesh.buffer().vertices_addr();
        }
        let vertex_buffers_buffer_info = vertex_buffers.descriptor_buffer_info(0);

        let vertex_megabuffer = Buffer::new(
            "Level vertex megabuffer",
            &engine.device,
            vertices_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            None,
        )?;
        let vertex_megabuffer_info = vertex_megabuffer.descriptor_buffer_info();

        let indices_size = (num_indices as usize * size_of::<u32>()) as vk::DeviceSize;
        let mut index_offsets_buffer = VolatileBuffer::<MeshBufferOffset, 1>::new_array(
            "Level index offsets",
            num_indices as usize,
            &engine.device,
            VolatileBufferType::Storage,
        )?;
        let index_offsets = meshes.iter().scan(0, |offset, mesh| {
            let result = *offset;
            *offset += mesh.num_indices();
            Some(result)
        });

        // Iterate over the ranges of indices for each mesh.
        let index_offsets_slice = index_offsets_buffer.get_mut_slice(0);
        for (i, (draw_offset, (start, end))) in izip!(
            mesh_draw_offsets.iter_mut(),
            index_offsets.chain(std::iter::once(num_indices)).tuple_windows(),
        )
        .enumerate()
        {
            draw_offset.index_offset = start;
            let index_offset = MeshBufferOffset {
                buffer_index: i as u32,
                offset: start,
            };
            index_offsets_slice[start as usize..end as usize].fill(index_offset);
        }
        let index_offsets_buffer_info = index_offsets_buffer.descriptor_buffer_info(0);

        let mut index_buffers = VolatileBuffer::<vk::DeviceAddress, 1>::new_array(
            "Level index buffer data",
            num_meshes,
            &engine.device,
            VolatileBufferType::Storage,
        )?;
        for (index_buffer_addr, mesh) in index_buffers.get_mut_slice(0).iter_mut().zip(meshes.iter()) {
            *index_buffer_addr = mesh.buffer().indices_addr();
        }
        let index_buffers_info = index_buffers.descriptor_buffer_info(0);
        let index_megabuffer = Buffer::new(
            "Level index megabuffer",
            &engine.device,
            indices_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            None,
        )?;
        let index_megabuffer_info = index_megabuffer.descriptor_buffer_info();

        // Write to descriptor set.
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

        // Run copy compute shaders.
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

        // Convert draw commands to hashmap.
        //let mesh_draw_commands = meshes.iter().cloned().zip(mesh_draw_commands.into_iter()).collect();
        Ok(Self {
            meshes,
            mesh_draw_offsets,
            _vertex_offsets_buffer: vertex_offsets_buffer,
            _vertex_buffers: vertex_buffers,
            vertex_megabuffer,
            _index_offsets_buffer: index_offsets_buffer,
            _index_buffers: index_buffers,
            index_megabuffer,
            _megabuffer_descriptor_pool: megabuffer_descriptor_pool,
        })
    }

    pub fn bind_megabuffer(&self, cmd_buf: &mut RecordingCmdBuf<PrimaryQueue, impl RenderingState>) {
        cmd_buf.bind_vertex_buffer(&self.vertex_megabuffer, 0);
        cmd_buf.bind_index_buffer(&self.index_megabuffer, 0, vk::IndexType::UINT32);
    }

    // TODO: Use mesh indices instead of arcs to prevent this linear search.
    #[deprecated]
    pub fn draw_command_for_mesh(&self, mesh: &Arc<Mesh>) -> &MeshDrawOffset {
        let index = self.meshes.iter().position(|test_mesh| Arc::ptr_eq(test_mesh, mesh));
        &self.mesh_draw_offsets[index.unwrap()]
    }
}
