use ash::vk;

use crate::device::{Device, SharedDeviceLoader};
use crate::gpu_alloc::MemoryError;
use crate::image::{Image, ImageDimensions};
use crate::pipeline::GraphicsShaderStageArray;
use crate::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};
use crate::utils;

#[derive(thiserror::Error, Debug)]
pub enum MaterialError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Material vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Material memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

pub struct Material {
    loader: SharedDeviceLoader,
    texture_image: Image,
    sampler: vk::Sampler,
    vertex_shader_module: ShaderModule<VertexShaderStage>,
    fragment_shader_module: ShaderModule<FragmentShaderStage>,
}

impl Material {
    pub fn new(device: &Device, texture_path: &str, gfx_cmd_pool: vk::CommandPool) -> Result<Self, MaterialError> {
        // Load texture.
        let image_file = std::fs::read(texture_path)?;
        let image = ktx2::Reader::new(image_file).unwrap();
        let header = image.header();
        let mip_levels = image.levels().collect::<Vec<_>>();
        let extent = vk::Extent2D { width: header.pixel_width, height: header.pixel_height };
        let texture_image = Image::new_from_mip_levels(
            "Model texture",
            device,
            gfx_cmd_pool,
            &mip_levels,
            ImageDimensions::Type2D(extent),
            header.format.map(utils::ktx_to_vulkan_format).unwrap_or(vk::Format::R8G8B8A8_SRGB),
        )?;

        // Create texture sampler.
        let max_anisotropy = device.get_properties().properties.limits.max_sampler_anisotropy;
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
            .max_lod(0.);
        let sampler = unsafe { device.loader().create_sampler(&sampler_info, None) }?;

        // Load shaders.
        let vertex_shader_code = std::fs::read("galaxy_engine/shaders/shader.vert.spv")?;
        let fragment_shader_code = std::fs::read("galaxy_engine/shaders/shader.frag.spv")?;

        let vertex_shader_module = ShaderModule::new(&device, &vertex_shader_code)?;
        let fragment_shader_module = ShaderModule::new(&device, &fragment_shader_code)?;

        Ok(Self {
            loader: device.cloned_loader(),
            texture_image,
            sampler,
            vertex_shader_module,
            fragment_shader_module,
        })
    }

    pub fn shader_stages(&self) -> GraphicsShaderStageArray {
        utils::arrayvec_from_array([
            self.vertex_shader_module.stage_info(),
            self.fragment_shader_module.stage_info(),
        ])
    }

    pub fn texture_image(&self) -> &Image {
        &self.texture_image
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn descriptor_set_layout_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        vec![ubo_layout_binding, sampler_layout_binding]
    }
}

impl Drop for Material {
    fn drop(&mut self) {
        // Drop sampler.
        unsafe { self.loader.destroy_sampler(self.sampler, None) };
    }
}
