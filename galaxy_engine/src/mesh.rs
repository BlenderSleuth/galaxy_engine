use std::fs::File;
use std::io::BufReader;
use std::slice;
use std::sync::Arc;

use ash::vk;
use meshopt::VertexDataAdapter;
use nalgebra as na;

use crate::buffer::{Buffer, GpuOnly};
use crate::descriptors::DescriptorPool;
use crate::device::{Device, QueueFamily, SharedDeviceLoader};
use crate::gpu_alloc::MemoryError;
use crate::material::{Material, MaterialError};
use crate::maths;
use crate::pipeline::{GraphicsPipeline, GraphicsPipelineParameters, Pipeline, PipelineLayout};
use crate::uniform_buffer::VolatileUniformBuffer;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    position: na::Vector3<f32>,
    color: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}

impl Vertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Obj error: {0}")]
    ObjError(#[from] obj::ObjError),
    #[error("Mesh vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Material error: {0}")]
    MaterialError(#[from] MaterialError),
}

pub struct Mesh {
    loader: SharedDeviceLoader,
    // TODO: Use a single buffer for both vertices and indices.
    vertex_buffer: Buffer<GpuOnly>,
    index_buffer: Buffer<GpuOnly>,
    _material: Material,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline: GraphicsPipeline,
    pub mvp: maths::ModelViewProjection,
}

impl Mesh {
    pub fn new(
        device: &Device,
        gfx_cmd_pool: vk::CommandPool,
        mesh_path: &str,
        texture_path: &str,
        samples: vk::SampleCountFlags,
        uniform_buffer: &VolatileUniformBuffer,
        descriptor_pool: &DescriptorPool,
    ) -> Result<Self, MeshError> {
        // Load material.
        let material = Material::new(device, texture_path, gfx_cmd_pool)?;

        // Load model. The obj crate already does indexing for us.
        let obj_model: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(BufReader::new(File::open(mesh_path)?))?;

        let vertices = obj_model
            .vertices
            .iter()
            .map(|v| Vertex {
                position: na::Vector3::new(v.position[0], v.position[1], v.position[2]),
                color: na::Vector3::new(1.0, 1.0, 1.0),
                tex_coord: na::Vector2::new(v.texture[0], 1.0 - v.texture[1]),
            })
            .collect::<Vec<Vertex>>();

        // Optimize model.
        let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vertices, Some(&obj_model.indices));
        let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &vert_remap);
        let mut indices = meshopt::remap_index_buffer(Some(&obj_model.indices), vertex_count, &vert_remap);
        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);
        let vertex_data_adapter = VertexDataAdapter::new(bytemuck::must_cast_slice(&vertices), std::mem::size_of::<Vertex>(), std::mem::offset_of!(Vertex, position)).unwrap();
        meshopt::optimize_overdraw_in_place(&mut indices, &vertex_data_adapter, 1.05);
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

        // Vertex buffer.
        let mut vertex_buffer = Buffer::<GpuOnly>::new_for_typed_data(
            "Mesh vertex buffer",
            &device,
            &vertices,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;
        vertex_buffer.copy_via_staging_buffer(&device, bytemuck::must_cast_slice(vertices.as_slice()), gfx_cmd_pool, QueueFamily::Graphics)?;

        // Index buffer.
        let mut index_buffer = Buffer::<GpuOnly>::new_for_typed_data(
            "Mesh index buffer",
            &device,
            &indices,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;
        index_buffer.copy_via_staging_buffer(&device, bytemuck::must_cast_slice(indices.as_slice()), gfx_cmd_pool, QueueFamily::Graphics)?;

        let layout_bindings = material.descriptor_set_layout_bindings();
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout = unsafe { device.loader().create_descriptor_set_layout(&descriptor_set_layout_info, None) }?;

        // Create push constant range.
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<maths::Mat4>() as u32);

        // Create pipeline layout.
        let pipeline_layout = Arc::new(PipelineLayout::new(&device, Some(&descriptor_set_layout), Some(&push_constant_range))?);

        // Create pipeline.
        let pipeline_params = GraphicsPipelineParameters {
            layout: pipeline_layout,
            vertex_binding_description: Vertex::binding_description(),
            vertex_attribute_descriptions: &Vertex::attribute_descriptions(),
            shader_stages: material.shader_stages(),
            samples,
            depth_test: true,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        // Create mesh descriptor sets.
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool.handle())
            .set_layouts(slice::from_ref(&descriptor_set_layout));
        let descriptor_set = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?[0];

        let buffer_info = uniform_buffer.descriptor_buffer_info();

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(material.texture_image().view().handle())
            .sampler(material.sampler());

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(slice::from_ref(&image_info)),
        ];

        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };
        Ok(Self {
            loader: device.cloned_loader(),
            vertex_buffer,
            index_buffer,
            _material: material,
            descriptor_set_layout,
            descriptor_set,
            pipeline,
            mvp: maths::ModelViewProjection::default(),
        })
    }

    //pub fn material(&self) -> &Material {
    //    &self.material
    //}

    //pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
    //    self.descriptor_set_layout
    //}

    pub fn record_graphics(&self, loader: &ash::Device, command_buffer: vk::CommandBuffer, viewport: vk::Viewport, scissor: vk::Rect2D) {
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle()) };
        unsafe { loader.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&self.vertex_buffer.handle()), slice::from_ref(&0)) };
        unsafe { loader.cmd_bind_index_buffer(command_buffer, self.index_buffer.handle(), 0, vk::IndexType::UINT32) };
        unsafe { loader.cmd_push_constants(command_buffer, self.pipeline.layout().handle(), vk::ShaderStageFlags::VERTEX, 0, bytemuck::cast_slice(&[self.mvp.mvp()])) };
        unsafe { loader.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout().handle(), 0, slice::from_ref(&self.descriptor_set), &[]) };
        unsafe { loader.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { loader.cmd_set_scissor(command_buffer, 0, &[scissor]) };
        unsafe { loader.cmd_draw_indexed(command_buffer, self.index_buffer.len(), 1, 0, 0, 0) };
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            //self.loader.free_descriptor_sets(self.descriptor_pool, slice::from_ref(&self.descriptor_set));
        }
    }
}