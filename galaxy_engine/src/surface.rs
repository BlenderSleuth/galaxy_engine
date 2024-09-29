use ash::{khr, vk};
use ash::prelude::VkResult;
use raw_window_handle::{DisplayHandle, WindowHandle};

pub struct Surface {
    loader: khr::surface::Instance,
    handle: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, display: DisplayHandle, window: WindowHandle) -> VkResult<Self> {
        Ok(Self {
            loader: khr::surface::Instance::new(entry, instance),
            handle: unsafe { ash_window::create_surface(&entry, &instance, display.as_raw(), window.as_raw(), None) }?,
        })
    }
    
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }
    
    pub fn get_physical_device_surface_support(&self, physical_device: vk::PhysicalDevice, queue_family_index: u32) -> VkResult<bool> {
        unsafe { self.loader.get_physical_device_surface_support(physical_device, queue_family_index, self.handle) }
    }
    
    pub fn get_present_modes(&self, physical_device: vk::PhysicalDevice) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe { self.loader.get_physical_device_surface_present_modes(physical_device, self.handle) }
    }

    pub fn get_capabilities(&self, physical_device: vk::PhysicalDevice) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe { self.loader.get_physical_device_surface_capabilities(physical_device, self.handle) }
    }
    
    pub fn get_formats(&self, physical_device: vk::PhysicalDevice) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe { self.loader.get_physical_device_surface_formats(physical_device, self.handle) }
    }
    
    pub unsafe fn destroy(&self) {
        unsafe { self.loader.destroy_surface(self.handle, None) };
    }
}