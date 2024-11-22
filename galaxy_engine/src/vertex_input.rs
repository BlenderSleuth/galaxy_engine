use ash::vk;
use ultraviolet::{Vec2, Vec3};

// For vertices with N attributes.
pub trait BindableVertex<const N: usize> {
    fn binding_description() -> vk::VertexInputBindingDescription;
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; N];
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PositionTexCoordVertex {
    pub position: Vec3,
    pub tex_coord: Vec2,
}

impl BindableVertex<2> for PositionTexCoordVertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<PositionTexCoordVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(PositionTexCoordVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(PositionTexCoordVertex, tex_coord) as u32),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ColouredVertex {
    pub position: Vec3,
    pub colour: Vec3,
    pub tex_coord: Vec2,
}

impl BindableVertex<3> for ColouredVertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ColouredVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, colour) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(ColouredVertex, tex_coord) as u32),
        ]
    }
}
