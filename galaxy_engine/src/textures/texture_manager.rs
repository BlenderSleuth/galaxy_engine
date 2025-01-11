// Copyright (c) 2024-2025 Ben Sutherland.

use ash::prelude::VkResult;
use ash::vk;
use indexmap::IndexMap;

use crate::engine::GalaxyEngine;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::textures::texture::TextureError;
use crate::textures::Texture;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::device::{Device, SharedDeviceLoader};

pub struct TextureManager {
    loader: SharedDeviceLoader,
    default_sampler: vk::Sampler,
    textures: IndexMap<ResourcePath, Texture>,
}

impl TextureManager {
    pub const MAX_TEXTURES: usize = 64;

    pub fn new(device: &Device) -> VkResult<Self> {
        // Create default texture sampler.
        let max_anisotropy = device.physical_device().properties.base.limits.max_sampler_anisotropy;
        let default_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(max_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(vk::LOD_CLAMP_NONE);
        let default_sampler = unsafe { device.loader().create_sampler(&default_sampler_info, None) }?;
        Ok(Self {
            loader: device.cloned_loader(),
            default_sampler,
            textures: IndexMap::new(),
        })
    }

    pub fn load_texture(
        &mut self,
        name: &str,
        path: &ResourcePath,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<u32, TextureError> {
        assert!(self.textures.len() < Self::MAX_TEXTURES);

        // Check if texture is already loaded.
        if let Some(texture_index) = self.textures.get_index_of(path) {
            return Ok(texture_index as u32);
        }

        // Load texture.
        let texture = Texture::new_from_ktx2_file(
            name,
            &path.full_path::<resource_type::Texture>(engine),
            &engine.device,
            cmd_pool,
        )?;
        let texture_index = self.textures.insert_full(path.clone(), texture).0;
        Ok(texture_index as u32)
    }

    pub fn num_textures(&self) -> u32 {
        self.textures.len() as u32
    }

    pub(crate) fn get_image_infos(&self) -> Vec<vk::DescriptorImageInfo> {
        self.textures
            .values()
            .map(|texture| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.image().view().handle())
                    .sampler(self.default_sampler)
            })
            .collect()
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_sampler(self.default_sampler, None);
        }
    }
}
