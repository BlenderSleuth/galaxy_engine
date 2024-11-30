// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;
use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;

use crate::engine::{GalaxyEngine, SceneDescriptorPool};
use crate::textures::texture::TextureError;
use crate::textures::Texture;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::image::Sampler;

pub struct TextureManager {
    loader: SharedDeviceLoader,
    default_sampler: Sampler,
    textures: Mutex<ArrayVec<Arc<Texture>, { TextureManager::NUM_TEXTURES }>>,
}

impl TextureManager {
    const NUM_TEXTURES: usize = 2;

    pub fn new(device: &Device) -> VkResult<Self> {
        // Create default texture sampler.
        let max_anisotropy = device.physical_device().properties.base.limits.max_sampler_anisotropy;
        let sampler_info = vk::SamplerCreateInfo::default()
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
        let default_sampler = Sampler::new(&device, &sampler_info)?;
        Ok(Self {
            loader: device.cloned_loader(),
            default_sampler,
            textures: Mutex::new(ArrayVec::new()),
        })
    }

    pub fn load_texture(
        &self,
        name: &str,
        path: &Path,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Arc<Texture>, TextureError> {
        let texture = Arc::new(Texture::new_from_ktx_file(name, path, device, cmd_pool)?);
        self.textures.lock().unwrap().push(texture.clone());
        Ok(texture)
    }

    pub fn write_textures_to_descriptor_array(
        &self,
        engine: &GalaxyEngine,
        scene_descriptor_pool: &SceneDescriptorPool,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.textures.lock().unwrap()[0].image().view().handle())
            .sampler(self.default_sampler.handle());
        // 2 of the same image.
        let image_infos = [image_info; Self::NUM_TEXTURES];

        let descriptor_writes: ArrayVec<_, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }> = scene_descriptor_pool
            .iter()
            .enumerate()
            .map(|(frame, set)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2) // Texture buffer is index 2 the in scene descriptor set layout.
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos)
            })
            .collect();

        unsafe { engine.device.loader().update_descriptor_sets(&descriptor_writes, &[]) };
    }
}
