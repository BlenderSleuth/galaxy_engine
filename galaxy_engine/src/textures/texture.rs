// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;

use ash::vk;

use crate::utils;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemoryError;
use crate::vulkan::image::{Image, ImageDimensions};

#[derive(Debug, thiserror::Error)]
pub enum TextureError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("KTX2 parse error: {0}")]
    Ktx2Error(#[from] ktx2::ParseError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

pub struct Texture {
    image: Image,
}

impl Texture {
    pub fn new_from_ktx2_file(
        name: &str,
        path: &Path,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, TextureError> {
        // Load texture.
        let texture_data = std::fs::read(path)?;
        let image = ktx2::Reader::new(texture_data)?;
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
