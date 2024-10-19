// Copyright (c) 2024. Ben Sutherland

pub mod debug;
pub mod device;
pub mod extensions;
pub mod instance;
pub mod physical_device;
pub mod queue;

use std::ffi::CStr;

use ash::vk;
pub use device::{get_device, Device, DeviceExt, SharedDeviceLoader};
pub use extensions::DeviceExtensions;
pub use instance::Instance;
pub use physical_device::{PhysicalDevice, PhysicalDeviceIncompatibility};
pub use queue::{queue_type, Queue};

pub const ENGINE_NAME: &'static CStr = c"Galaxy Engine";
pub const ENGINE_VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");
pub const MIN_VK_VERSION: u32 = ash::vk::make_api_version(0, 1, 2, 0);

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
