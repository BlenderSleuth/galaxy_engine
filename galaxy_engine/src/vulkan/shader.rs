// Copyright (c) 2024-2025 Ben Sutherland.

use std::path::Path;

use ash::prelude::VkResult;
use ash::vk;

use crate::pipelines::PipelineManager;
use crate::vulkan::device::{Device, SharedDeviceLoader};

pub mod shader_stage {
    use ash::vk;

    // Shader stage type state pattern.
    pub trait ShaderStageType {
        fn stage() -> vk::ShaderStageFlags;
    }
    pub struct Vertex;
    impl ShaderStageType for Vertex {
        fn stage() -> vk::ShaderStageFlags {
            vk::ShaderStageFlags::VERTEX
        }
    }
    pub struct Fragment;
    impl ShaderStageType for Fragment {
        fn stage() -> vk::ShaderStageFlags {
            vk::ShaderStageFlags::FRAGMENT
        }
    }
    pub struct Compute;
    impl ShaderStageType for Compute {
        fn stage() -> vk::ShaderStageFlags {
            vk::ShaderStageFlags::COMPUTE
        }
    }
}
use shader_stage::*;

pub struct ShaderModule<S: ShaderStageType> {
    loader: SharedDeviceLoader,
    handle: vk::ShaderModule,
    _marker: std::marker::PhantomData<S>,
}

impl<S: ShaderStageType> ShaderModule<S> {
    pub fn new(device: &Device, config_path: &str) -> VkResult<Self> {
        let path = Path::new(PipelineManager::BUILT_SHADER_PATH)
            .join(config_path)
            .with_extension("spv");

        // We unwrap here so we can provide a more informative error message for invalid shaders.
        // Ash utility function handles code alignment and endianness.
        let code = ash::util::read_spv(
            &mut std::fs::File::open(&path).unwrap_or_else(|err| panic!("Invalid shader path: {path:?} ({err}).")),
        )
        .unwrap_or_else(|err| panic!("Invalid shader file: {path:?} ({err})."));

        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        Ok(Self {
            loader: device.cloned_loader(),
            handle: unsafe { device.loader.create_shader_module(&create_info, None) }?,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn stage_info(&self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage())
            .module(self.handle)
            .name(match self.stage() {
                vk::ShaderStageFlags::VERTEX => c"mainVS",
                vk::ShaderStageFlags::FRAGMENT => c"mainFS",
                vk::ShaderStageFlags::COMPUTE => c"mainCS",
                _ => panic!("Unsupported shader stage: {:?}", self.stage()),
            })
    }

    pub fn stage(&self) -> vk::ShaderStageFlags {
        S::stage()
    }
}

impl<S: ShaderStageType> Drop for ShaderModule<S> {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_shader_module(self.handle, None) };
    }
}
