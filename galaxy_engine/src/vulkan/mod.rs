// Copyright (c) 2024-2025 Ben Sutherland.

pub mod buffer;
pub mod command_buffer;
pub mod debug;
pub mod descriptors;
pub mod device;
pub mod extensions;
pub mod gpu_alloc;
pub mod image;
pub mod instance;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;

use std::ffi::CStr;

use ash::vk;
pub use device::physical_device;
pub use device::queue;
pub use device::get_device_loader;

pub const ENGINE_NAME: &CStr = c"Galaxy Engine";
pub const ENGINE_VERSION_STR: &str = env!("CARGO_PKG_VERSION");
pub const ENGINE_VERSION: u32 = crate::utils::pkg_version();
pub const MIN_VK_VERSION: u32 = vk::make_api_version(0, 1, 2, 0);

#[derive(Debug, thiserror::Error)]
#[error("App requires Vulkan {}.{}.{} (Current: {}.{}.{}). Consider updating your graphics drivers",
    vk::api_version_major(MIN_VK_VERSION),
    vk::api_version_minor(MIN_VK_VERSION),
    vk::api_version_patch(MIN_VK_VERSION),
    vk::api_version_major(*.api_version),
    vk::api_version_minor(*.api_version),
    vk::api_version_patch(*.api_version)
)]
pub struct IncompatibleVulkanVersion {
    pub api_version: u32,
}

impl From<u32> for IncompatibleVulkanVersion {
    fn from(api_version: u32) -> Self {
        Self { api_version }
    }
}
