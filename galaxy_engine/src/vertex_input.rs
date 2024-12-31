// Copyright (c) 2024-2025 Ben Sutherland.

use std::slice;

use ash::vk;
use ultraviolet::{Vec2, Vec3, Vec4};

pub const fn binding_description_for_type<T>() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<T>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }
}

// For vertices with N attributes. TODO: Generate with a macro.
pub trait BindableVertex<const N: usize>: bytemuck::Pod {
    fn binding_description() -> &'static vk::VertexInputBindingDescription;
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; N];
    fn vertex_input_state() -> vk::PipelineVertexInputStateCreateInfo<'static> {
        vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(Self::binding_description()))
            .vertex_attribute_descriptions(Self::attribute_descriptions())
    }
}

#[derive(serde::Deserialize, Debug)]
pub enum VertexInputType {
    PositionTexCoord,
    Mesh,
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PositionTexCoordVertex {
    pub position: Vec3,
    pub tex_coord: Vec2,
}

impl BindableVertex<2> for PositionTexCoordVertex {
    fn binding_description() -> &'static vk::VertexInputBindingDescription {
        const DESCRIPTION: vk::VertexInputBindingDescription = binding_description_for_type::<MeshVertex>();
        &DESCRIPTION
    }
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; 2] {
        const DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(PositionTexCoordVertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32_SFLOAT,
                offset: std::mem::offset_of!(PositionTexCoordVertex, tex_coord) as u32,
            },
        ];
        &DESCRIPTIONS
    }
}
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct MeshVertex {
    pub position: Vec3,
    pub tex_coord_u: f32,
    pub normal: Vec3,
    pub tex_coord_v: f32,
    pub tangent: Vec4,
}

impl BindableVertex<5> for MeshVertex {
    fn binding_description() -> &'static vk::VertexInputBindingDescription {
        const DESCRIPTION: vk::VertexInputBindingDescription = binding_description_for_type::<MeshVertex>();
        &DESCRIPTION
    }
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; 5] {
        const DESCRIPTIONS: [vk::VertexInputAttributeDescription; 5] = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(MeshVertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32_SFLOAT,
                offset: std::mem::offset_of!(MeshVertex, tex_coord_u) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(MeshVertex, normal) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                format: vk::Format::R32_SFLOAT,
                offset: std::mem::offset_of!(MeshVertex, tex_coord_v) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: std::mem::offset_of!(MeshVertex, tangent) as u32,
            },
        ];
        &DESCRIPTIONS
    }
}
