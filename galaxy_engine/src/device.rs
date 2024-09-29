use ash::{khr, vk};
use crate::surface::Surface;
use crate::utils;

#[derive(Debug, thiserror::Error)]
pub enum DeviceInitError {
    #[error("No physical devices found.")]
    NoPhysicalDevices,
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
}

#[derive(Debug, Clone)]
pub struct PhysicalDeviceProperties {
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue_family_idx: u32,
    pub present_queue_family_idx: u32,
    pub is_discrete: bool,
    pub swapchain_format: vk::SurfaceFormatKHR,
    pub presentation_mode: vk::PresentModeKHR,
    pub image_count: u32,
}

impl PhysicalDeviceProperties {
    const DEPTH_STENCIL_FORMAT: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;

    pub fn get_unique_queue_families(&self) -> Vec<u32> {
        let mut unique_queue_families = vec![self.graphics_queue_family_idx, self.present_queue_family_idx];
        unique_queue_families.sort_unstable();
        unique_queue_families.dedup();
        unique_queue_families
    }
}

pub struct Device {
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    properties: PhysicalDeviceProperties,
}

impl Device {
    pub fn new(instance: &ash::Instance, surface: &Surface) -> Result<Self, DeviceInitError> {
        // Pick physical device.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        if physical_devices.is_empty() {
            return Err(DeviceInitError::NoPhysicalDevices);
        }

        let required_device_extensions = &[khr::swapchain::NAME, khr::synchronization2::NAME, khr::dynamic_rendering::NAME];

        let mut current_device_properties = None;
        for physical_device in physical_devices.iter() {
            // Check device extensions.
            let available_extensions = unsafe { instance.enumerate_device_extension_properties(*physical_device) }?;

            let mut has_required_extensions = true;
            for required_extension in required_device_extensions.iter() {
                if !available_extensions.iter().any(|&available_extension| {
                    available_extension.extension_name_as_c_str() == Ok(required_extension)
                }) {
                    log::warn!("Required extension not found: {:?}", required_extension);
                    has_required_extensions = false;
                    break;
                }
            }
            if !has_required_extensions {
                continue;
            }

            // Select queue families.
            let mut graphics_queue_family_idx = None;
            let mut present_queue_family_idx = None;
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

            for (queue_family_idx, queue_family) in queue_families.iter().enumerate() {
                let queue_family_idx = queue_family_idx as u32;
                let is_present_supported = surface.get_physical_device_surface_support(*physical_device, queue_family_idx)?;

                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    // Prefer queue family that supports both graphics and present.
                    if is_present_supported {
                        graphics_queue_family_idx = Some(queue_family_idx);
                        present_queue_family_idx = Some(queue_family_idx);
                    } else if graphics_queue_family_idx.is_none() {
                        graphics_queue_family_idx = Some(queue_family_idx);
                    }
                } else if present_queue_family_idx.is_none() && is_present_supported {
                    // Present-only queue family.
                    present_queue_family_idx = Some(queue_family_idx);
                }
            }

            if graphics_queue_family_idx.is_none() || present_queue_family_idx.is_none() {
                continue;
            }

            // Require VK_FORMAT_D32_SFLOAT_S8_UINT for depth/stencil.
            let format_properties = unsafe { instance.get_physical_device_format_properties(*physical_device, PhysicalDeviceProperties::DEPTH_STENCIL_FORMAT) };
            if !format_properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
                continue;
            }

            // Require compatible surface properties.
            let surface_capabilities = surface.get_capabilities(*physical_device)?; 
            let surface_formats = surface.get_formats(*physical_device)?;
            let surface_present_modes = surface.get_present_modes(*physical_device)?;

            if surface_formats.is_empty() || surface_present_modes.is_empty() {
                continue;
            }

            // Choose swapchain format and present mode.
            let Some(swapchain_format) = surface_formats.into_iter().find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            }) else {
                continue;
            };
            let Some(presentation_mode) = surface_present_modes.into_iter().find(|&mode| mode == vk::PresentModeKHR::MAILBOX) else {
                continue;
            };

            let mut image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0 && image_count > surface_capabilities.max_image_count {
                image_count = surface_capabilities.max_image_count;
            }

            let physical_device_properties = unsafe { instance.get_physical_device_properties(*physical_device) };
            
            let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
            let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut dynamic_rendering_features);
            unsafe { instance.get_physical_device_features2(*physical_device, &mut physical_device_features) };
            
            // Require dynamic rendering support.
            if dynamic_rendering_features.dynamic_rendering == vk::FALSE {
                continue;
            }
            
            let device_properties = PhysicalDeviceProperties {
                physical_device: *physical_device,
                graphics_queue_family_idx: graphics_queue_family_idx.unwrap(),
                present_queue_family_idx: present_queue_family_idx.unwrap(),
                is_discrete: physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                swapchain_format,
                presentation_mode,
                image_count,
            };

            // Prefer discrete GPU.
            if current_device_properties.is_none() || device_properties.is_discrete {
                current_device_properties = Some(device_properties);
            }
        }

        let Some(current_device_properties) = current_device_properties else {
            return Err(DeviceInitError::NoPhysicalDevices);
        };

        let unique_queue_families = current_device_properties.get_unique_queue_families();

        // Create logical device.
        let mut queue_infos = Vec::with_capacity(unique_queue_families.len());
        for unique_queue_family in unique_queue_families.iter() {
            queue_infos.push(vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*unique_queue_family)
                .queue_priorities(&[1.0]));
        }

        // Enable dynamic rendering.
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default()
            .dynamic_rendering(true);
        
        // Enable synchronization2.
        let mut synchronization2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);
        
        let device_features = vk::PhysicalDeviceFeatures::default();
        let device_extensions = utils::cstr_to_ptrs(required_device_extensions);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut synchronization2_features);

        let device = unsafe { instance.create_device(current_device_properties.physical_device, &device_info, None) }?;

        // Get queues.
        let graphics_queue = unsafe { device.get_device_queue(current_device_properties.graphics_queue_family_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(current_device_properties.present_queue_family_idx, 0) };

        Ok(Self { device, graphics_queue, present_queue, properties: current_device_properties })
    }

    pub fn get_properties(&self) -> &PhysicalDeviceProperties {
        &self.properties
    }
    
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }
    
    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }
    
    pub fn device(&self) -> &ash::Device {
        &self.device
    }
    
    pub unsafe fn destroy(&self) {
        unsafe { self.device.destroy_device(None) };
    }
}
