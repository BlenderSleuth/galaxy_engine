// Copyright (c) 2024 Ben Sutherland.

use std::ffi::CStr;

#[cfg(feature = "debug_info")]
use ash::prelude::VkResult;
use ash::{ext, khr};

pub struct DeviceExtensions {
    #[cfg(feature = "debug_info")]
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    pub sync2: khr::synchronization2::Device,
    pub dyn_cmd: khr::dynamic_rendering::Device,
    pub desc_buf: ext::descriptor_buffer::Device,
}

impl DeviceExtensions {
    pub(super) fn new(instance: &ash::Instance, device: &ash::Device, _optional_extensions: &[&CStr]) -> Self {
        #[cfg(feature = "debug_info")]
        let debug_utils = if _optional_extensions.iter().any(|&ext| ext == ext::debug_utils::NAME) {
            Some(ext::debug_utils::Device::new(&instance, &device))
        } else {
            None
        };
        let sync2 = khr::synchronization2::Device::new(&instance, &device);
        let dyn_cmd = khr::dynamic_rendering::Device::new(&instance, &device);
        let desc_buf = ext::descriptor_buffer::Device::new(&instance, &device);
        Self {
            #[cfg(feature = "debug_info")]
            debug_utils,
            sync2,
            dyn_cmd,
            desc_buf,
        }
    }

    #[cfg(feature = "debug_info")]
    pub fn run_debug(&self, f: impl FnOnce(&ash::ext::debug_utils::Device) -> VkResult<()>) -> VkResult<()> {
        if let Some(debug) = &self.debug_utils {
            f(debug)
        } else {
            Ok(())
        }
    }
}
