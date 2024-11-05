// Copyright (c) 2024 Ben Sutherland.

use ash::vk;

use crate::utils;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::image::{Image, ImageDimensions};

pub struct Texture {
    image: Image,
}

impl Texture {
    pub fn new_from_file(
        name: &str,
        path: &str,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        // Load texture.
        let texture_file = std::fs::read(path).unwrap();
        let image = ktx2::Reader::new(texture_file).unwrap();
        let header = image.header();
        let mip_levels = image.levels().collect::<Vec<_>>();
        let extent = vk::Extent2D {
            width: header.pixel_width,
            height: header.pixel_height,
        };
        let texture_image = Image::new_from_mip_levels(
            debug::debug_only_name!("{name} texture"),
            device,
            cmd_pool,
            &mip_levels,
            ImageDimensions::Type2D(extent),
            header.format.map(utils::ktx_to_vulkan_format).unwrap(),
        )?;

        Ok(Self { image: texture_image })
    }

    pub fn image(&self) -> &Image {
        &self.image
    }
}
