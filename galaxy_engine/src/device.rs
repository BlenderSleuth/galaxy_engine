use std::cell::RefCell;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::Arc;
use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::{khr, vk};
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use itertools::Itertools;

use crate::engine::MemResult;
use crate::surface::Surface;
use crate::utils;

#[derive(Debug, thiserror::Error)]
pub enum DeviceInitError {
    #[error("No physical devices found.")]
    NoPhysicalDevices,
    #[error("No compatible physical devices found.")]
    NoCompatiblePhysicalDevices,
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Allocator error: {0}")]
    AllocatorError(#[from] gpu_allocator::AllocationError),
}

pub struct LoadedExtensions {
    pub sync2: khr::synchronization2::Device,
    pub dyn_cmd: khr::dynamic_rendering::Device,
}

impl LoadedExtensions {
    fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        let sync2 = khr::synchronization2::Device::new(&instance, &device);
        let dyn_cmd = khr::dynamic_rendering::Device::new(&instance, &device);
        Self { sync2, dyn_cmd }
    }
}

pub enum QueueFamily {
    Graphics,
    Present,
    Transfer,
}

#[derive(Debug, Clone)]
pub struct PhysicalDeviceProperties {
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue_family_idx: u32,
    pub present_queue_family_idx: u32,
    pub transfer_queue_family_idx: u32,
    pub is_discrete: bool,
    pub swapchain_format: vk::SurfaceFormatKHR,
    pub presentation_mode: vk::PresentModeKHR,
    pub image_count: u32,
    pub properties: vk::PhysicalDeviceProperties,
}

//noinspection RsUnresolvedPath
pub type PropertyQueueList = ArrayVec<u32, { PhysicalDeviceProperties::MAX_QUEUE_FAMILIES }>;
impl PhysicalDeviceProperties {
    pub(crate) const DEPTH_STENCIL_FORMAT: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;
    pub(crate) const MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_8;
    const MAX_QUEUE_FAMILIES: usize = 3;

    pub fn get_unique_queue_families(&self) -> PropertyQueueList {
        let mut unique_queue_families = PropertyQueueList::from([self.graphics_queue_family_idx, self.present_queue_family_idx, self.transfer_queue_family_idx]);
        unique_queue_families.sort_unstable();
        let mut result = PropertyQueueList::new();
        result.extend(unique_queue_families.into_iter().dedup());
        result
    }
}

pub struct Device {
    device: ManuallyDrop<Arc<ash::Device>>,
    ext: LoadedExtensions,
    allocator: RefCell<ManuallyDrop<Allocator>>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    transfer_queue: vk::Queue,
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

            // Require graphics and present queue families.
            let (Some(graphics_queue_family_idx), Some(present_queue_family_idx)) =
                (graphics_queue_family_idx, present_queue_family_idx) else {
                continue;
            };

            // Find separate transfer queue family. 
            // Default to graphics queue family, which implicitly supports transfer operations.
            let mut transfer_queue_family_idx = graphics_queue_family_idx;
            for (queue_family_idx, queue_family) in queue_families.iter().enumerate() {
                // Choose first queue family that supports transfer operations but is not a graphics queue.
                let queue_family_idx = queue_family_idx as u32;
                if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) && !queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    transfer_queue_family_idx = queue_family_idx;
                    break;
                }
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

            // Calculate swapchain image count.
            let mut image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0 && image_count > surface_capabilities.max_image_count {
                image_count = surface_capabilities.max_image_count;
            }

            let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
            unsafe { instance.get_physical_device_properties2(*physical_device, &mut physical_device_properties) };
            let physical_device_properties = physical_device_properties.properties;

            // Require 8 MSAA samples.
            if !physical_device_properties.limits.framebuffer_color_sample_counts.contains(PhysicalDeviceProperties::MSAA_SAMPLES) {
                continue;
            }

            let mut buffer_device_address_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut buffer_device_address_features);
            unsafe { instance.get_physical_device_features2(*physical_device, &mut physical_device_features) };

            // Require anisotropic filtering support.
            if physical_device_features.features.sampler_anisotropy == vk::FALSE {
                continue;
            }

            // Require buffer_device_address support.
            if buffer_device_address_features.buffer_device_address == vk::FALSE {
                continue;
            }

            let device_properties = PhysicalDeviceProperties {
                physical_device: *physical_device,
                graphics_queue_family_idx,
                present_queue_family_idx,
                transfer_queue_family_idx,
                is_discrete: physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                swapchain_format,
                presentation_mode,
                image_count,
                properties: physical_device_properties,
            };

            // Prefer discrete GPU.
            if current_device_properties.is_none() || device_properties.is_discrete {
                current_device_properties = Some(device_properties);
            }
        }

        let Some(current_device_properties) = current_device_properties else {
            return Err(DeviceInitError::NoCompatiblePhysicalDevices);
        };

        let unique_queue_families = current_device_properties.get_unique_queue_families();

        // Create logical device.
        let mut queue_infos = Vec::with_capacity(unique_queue_families.len());
        for unique_queue_family in unique_queue_families.iter() {
            queue_infos.push(vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*unique_queue_family)
                .queue_priorities(&[1.0])
            );
        }

        // Enable dynamic rendering.
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default()
            .dynamic_rendering(true);

        // Enable synchronization2.
        let mut synchronization2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);

        let device_features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true);
        let mut device_features_12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true);

        let device_extensions = utils::cstr_to_ptrs(required_device_extensions);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut synchronization2_features)
            .push_next(&mut device_features_12);

        let device = unsafe { instance.create_device(current_device_properties.physical_device, &device_info, None) }?;

        // Get queues.
        let graphics_queue = unsafe { device.get_device_queue(current_device_properties.graphics_queue_family_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(current_device_properties.present_queue_family_idx, 0) };
        let transfer_queue = unsafe { device.get_device_queue(current_device_properties.transfer_queue_family_idx, 0) };

        // TODO: This allocator keeps a copy of the device and instance, which is not ideal.
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: current_device_properties.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,  // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        })?;

        // Load extensions.
        let ext = LoadedExtensions::new(&instance, &device);

        Ok(Self {
            device: ManuallyDrop::new(Arc::new(device)),
            ext,
            allocator: RefCell::new(ManuallyDrop::new(allocator)),
            graphics_queue,
            present_queue,
            transfer_queue,
            properties: current_device_properties,
        })
    }

    pub fn device(&self) -> &Arc<ash::Device> {
        &self.device
    }

    pub fn ext(&self) -> &LoadedExtensions {
        &self.ext
    }

    pub fn get_properties(&self) -> &PhysicalDeviceProperties {
        &self.properties
    }

    pub fn get_queue(&self, queue: QueueFamily) -> vk::Queue {
        match queue {
            QueueFamily::Graphics => self.graphics_queue,
            QueueFamily::Present => self.present_queue,
            QueueFamily::Transfer => self.transfer_queue,
        }
    }
    pub fn get_queue_family_idx(&self, queue: QueueFamily) -> u32 {
        match queue {
            QueueFamily::Graphics => self.properties.graphics_queue_family_idx,
            QueueFamily::Present => self.properties.present_queue_family_idx,
            QueueFamily::Transfer => self.properties.transfer_queue_family_idx,
        }
    }

    pub fn allocate_memory(&self, desc: &AllocationCreateDesc) -> MemResult<gpu_allocator::vulkan::Allocation> {
        Ok(self.allocator.borrow_mut().allocate(desc)?)
    }

    pub fn free_memory(&self, allocation: gpu_allocator::vulkan::Allocation) -> MemResult<()> {
        Ok(self.allocator.borrow_mut().free(allocation)?)
    }

    pub fn print_allocator_report(&self) {
        log::info!("{:?}", self.allocator.borrow().generate_report());
    }

    // Doesn't allocate a vec for the single buffer case.
    pub unsafe fn allocate_command_buffer(&self, cmd_pool: vk::CommandPool, level: vk::CommandBufferLevel) -> VkResult<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(level)
            .command_buffer_count(1);
        let mut buffer = MaybeUninit::uninit();
        (self.device.fp_v1_0().allocate_command_buffers)(
            self.device.handle(),
            &allocate_info,
            buffer.as_mut_ptr(),
        ).result()?;
        Ok(buffer.assume_init())
    }

    pub fn create_transient_command_pool(&self, queue_family: QueueFamily) -> VkResult<vk::CommandPool> {
        // Create command pool.
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.get_queue_family_idx(queue_family));
        unsafe { self.device.create_command_pool(&command_pool_info, None) }
    }

}

impl Drop for Device {
    fn drop(&mut self) {
        // Drop allocator.
        unsafe { ManuallyDrop::drop(self.allocator.get_mut()) };
        // Drop device. Ensures this is the last reference to the device.
        unsafe { Arc::into_inner(ManuallyDrop::take(&mut self.device)).unwrap().destroy_device(None) };
    }
}

//pub struct SharedDevice(Arc<ManuallyDrop<Device>>);
//
//impl SharedDevice {
//    pub fn new(device: Device) -> Self {
//        Self(Arc::new(ManuallyDrop::new(device)))
//    }
//
//    pub fn destroy(&mut self) {
//        unsafe { ManuallyDrop::drop(&mut Arc::into_inner(self.0)) };
//    }
//}
//
//impl std::ops::Deref for SharedDevice {
//    type Target = Device;
//
//    fn deref(&self) -> &Self::Target {
//        &self.0
//    }
//}
