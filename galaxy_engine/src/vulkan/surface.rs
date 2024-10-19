// Copyright (c) 2024. Ben Sutherland

use ash::prelude::VkResult;
use ash::{khr, vk};
use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::vulkan::instance::Instance;

pub struct Surface {
    loader: khr::surface::Instance,
    handle: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, display: DisplayHandle, window: WindowHandle) -> VkResult<Self> {
        Ok(Self {
            loader: khr::surface::Instance::new(instance.entry(), instance.loader()),
            handle: unsafe {
                ash_window::create_surface(
                    instance.entry(),
                    instance.loader(),
                    display.as_raw(),
                    window.as_raw(),
                    None,
                )
            }?,
        })
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    pub fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<bool> {
        unsafe {
            self.loader
                .get_physical_device_surface_support(physical_device, queue_family_index, self.handle)
        }
    }

    pub fn get_present_modes(&self, physical_device: vk::PhysicalDevice) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.handle)
        }
    }

    pub fn get_capabilities(&self, physical_device: vk::PhysicalDevice) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.handle)
        }
    }

    pub fn get_formats(&self, physical_device: vk::PhysicalDevice) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.handle)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.handle, None) };
    }
}
