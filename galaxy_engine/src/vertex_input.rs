// Copyright (c) 2024 Ben Sutherland.

use std::slice;

use ash::vk;
use ultraviolet::{Vec2, Vec3};

pub const fn binding_description_for_type<T>() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<T>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }
}

// For vertices with N attributes. TODO: Generate with a macro.
pub trait BindableVertex<const N: usize> {
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
    PositionColourTexCoord,
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PositionTexCoordVertex {
    pub position: Vec3,
    pub element_index: u32,
    pub tex_coord: Vec2,
}

impl BindableVertex<3> for PositionTexCoordVertex {
    fn binding_description() -> &'static vk::VertexInputBindingDescription {
        const DESCRIPTION: vk::VertexInputBindingDescription = binding_description_for_type::<PositionTexCoordVertex>();
        &DESCRIPTION
    }
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; 3] {
        const DESCRIPTIONS: [vk::VertexInputAttributeDescription; 3] = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(PositionTexCoordVertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32_UINT,
                offset: std::mem::offset_of!(PositionTexCoordVertex, element_index) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: std::mem::offset_of!(PositionTexCoordVertex, tex_coord) as u32,
            },
        ];
        &DESCRIPTIONS
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PositionColourTexCoordVertex {
    pub position: Vec3,
    pub colour: Vec3,
    pub tex_coord: Vec2,
}

impl BindableVertex<3> for PositionColourTexCoordVertex {
    fn binding_description() -> &'static vk::VertexInputBindingDescription {
        static DESCRIPTION: vk::VertexInputBindingDescription =
            binding_description_for_type::<PositionColourTexCoordVertex>();
        &DESCRIPTION
    }
    fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription; 3] {
        static DESCRIPTIONS: [vk::VertexInputAttributeDescription; 3] = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(PositionColourTexCoordVertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(PositionColourTexCoordVertex, colour) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: std::mem::offset_of!(PositionColourTexCoordVertex, tex_coord) as u32,
            },
        ];
        &DESCRIPTIONS
    }
}
