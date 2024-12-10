// Copyright (c) 2024 Ben Sutherland.

use ash::vk;

pub const DEFAULT_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub fn get_aspect_for_format(format: vk::Format) -> vk::ImageAspectFlags {
    use vk::Format;
    match format {
        Format::D16_UNORM | Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}
