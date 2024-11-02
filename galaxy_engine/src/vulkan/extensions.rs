// Copyright (c) 2024 Ben Sutherland.

use std::ffi::CStr;

use ash::khr;
#[cfg(feature = "debug_info")]
use ash::prelude::VkResult;

pub struct DeviceExtensions {
    #[cfg(feature = "debug_info")]
    pub debug: Option<ash::ext::debug_utils::Device>,
    pub sync2: khr::synchronization2::Device,
    pub dyn_cmd: khr::dynamic_rendering::Device,
}

impl DeviceExtensions {
    pub(super) fn new(instance: &ash::Instance, device: &ash::Device, _optional_extensions: &[&CStr]) -> Self {
        #[cfg(feature = "debug_info")]
        let debug = if _optional_extensions
            .iter()
            .any(|&ext| ext == ash::ext::debug_utils::NAME)
        {
            Some(ash::ext::debug_utils::Device::new(&instance, &device))
        } else {
            None
        };
        let sync2 = khr::synchronization2::Device::new(&instance, &device);
        let dyn_cmd = khr::dynamic_rendering::Device::new(&instance, &device);
        Self {
            #[cfg(feature = "debug_info")]
            debug,
            sync2,
            dyn_cmd,
        }
    }

    #[cfg(feature = "debug_info")]
    pub fn run_debug(&self, f: impl FnOnce(&ash::ext::debug_utils::Device) -> VkResult<()>) -> VkResult<()> {
        if let Some(debug) = &self.debug {
            f(debug)
        } else {
            Ok(())
        }
    }
}
