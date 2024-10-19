// Copyright (c) 2024. Ben Sutherland

use std::ffi::CStr;

use arrayvec::ArrayVec;
use ash::vk;

use crate::vulkan;
use crate::vulkan::surface::Surface;

#[derive(Debug, thiserror::Error)]
pub enum PhysicalDeviceIncompatibility {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Required extensions not implemented: {0:?}")]
    RequiredExtensionsNotImplemented(Vec<&'static CStr>),
    #[error("No primary queue family found")]
    NoPrimaryQueueFamily,
    #[error("No compatible depth/stencil format found: {0:?}")]
    NoDepthStencilFormat(vk::Format),
    #[error("No compatible surface formats found: {0:?}")]
    NoCompatibleSurfaceFormats(Vec<vk::SurfaceFormatKHR>),
    #[error("Incompatible Vulkan version: {0}")]
    IncompatibleVulkanVersion(vulkan::IncompatibleVulkanVersion),
    #[error("Not enough push constant space: {0} < {1}")]
    NotEnoughPushConstantSpace(u32, u32),
    #[error("No anisotropic filtering support")]
    NoAnisotropicFiltering,
    #[error("No buffer device address support")]
    NoBufferDeviceAddress,
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub primary_queue_family_idx: u32,
    pub async_transfer_queue_family_idx: Option<u32>,
    pub async_compute_queue_family_idx: Option<u32>,
    pub is_discrete: bool,
    pub swapchain_format: vk::SurfaceFormatKHR,
    pub presentation_mode: vk::PresentModeKHR,
    pub depth_stencil_format: vk::Format,
    pub swapchain_image_count: u32,
    pub supported_msaa_samples: vk::SampleCountFlags,
    pub max_msaa_samples: vk::SampleCountFlags,
    pub properties: vk::PhysicalDeviceProperties,
}

//noinspection RsUnresolvedPath
pub type PropertyQueueList = ArrayVec<u32, { PhysicalDevice::MAX_QUEUE_FAMILIES }>;
impl PhysicalDevice {
    const MAX_QUEUE_FAMILIES: usize = 3;

    pub fn new(
        instance: &ash::Instance,
        surface: &Surface,
        handle: vk::PhysicalDevice,
        required_device_extensions: &[&'static CStr],
    ) -> Result<PhysicalDevice, PhysicalDeviceIncompatibility> {
        // Check vulkan extensions.
        let available_extensions = unsafe { instance.enumerate_device_extension_properties(handle) }?;

        let mut not_implemented_extensions = required_device_extensions.to_vec();
        not_implemented_extensions.retain(|&required_extension| {
            // Retain all that are not available.
            !available_extensions
                .iter()
                .any(|&available_extension| available_extension.extension_name_as_c_str() == Ok(required_extension))
        });

        if !not_implemented_extensions.is_empty() {
            return Err(PhysicalDeviceIncompatibility::RequiredExtensionsNotImplemented(
                not_implemented_extensions,
            ));
        }

        // Select queue families.
        let mut primary_queue_family_idx = None;
        let mut async_transfer_queue_family_idx = None;
        let mut async_compute_queue_family_idx = None;
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(handle) };

        for (queue_family_idx, queue_family) in queue_families.iter().enumerate() {
            let queue_family_idx = queue_family_idx as u32;

            // Generally, choose the first queue family that fits the requirements.
            let queue_flags = queue_family.queue_flags;
            let supports_graphics = queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_compute = queue_flags.contains(vk::QueueFlags::COMPUTE);
            // Note: Queues that support compute/graphics ops implicitly support transfer ops, but don't necessarily report that they do.
            let supports_transfer = queue_flags.contains(vk::QueueFlags::TRANSFER);
            // Async transfer queue can get muddled into video codec queues, so we need to check for that.
            let supports_video_codec =
                queue_flags.intersects(vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::VIDEO_ENCODE_KHR);
            let supports_present = surface.get_physical_device_surface_support(handle, queue_family_idx)?;

            // This queue family is assigned to at most one of the following roles:
            // Find the primary queue.
            if primary_queue_family_idx.is_none() && supports_graphics && supports_compute && supports_present {
                primary_queue_family_idx = Some(queue_family_idx);
            // Find a queue family that supports only compute operations.
            } else if async_compute_queue_family_idx.is_none() && supports_compute && !supports_graphics {
                async_compute_queue_family_idx = Some(queue_family_idx);
            // Find a queue family that supports only transfer operations.
            } else if async_compute_queue_family_idx.is_none()
                && supports_transfer
                && !supports_graphics
                && !supports_compute
                && !supports_video_codec
            {
                async_transfer_queue_family_idx = Some(queue_family_idx);
            }
        }

        // Require the primary queue family.
        let primary_queue_family_idx =
            primary_queue_family_idx.ok_or(PhysicalDeviceIncompatibility::NoPrimaryQueueFamily)?;

        // Require specific format for depth/stencil.
        // Nvidia recommends 24-bit depth buffer with 8-bit stencil buffer, but AMD recommends 32-bit float depth buffer.
        const DEPTH_STENCIL_FORMAT: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;
        let format_properties = unsafe { instance.get_physical_device_format_properties(handle, DEPTH_STENCIL_FORMAT) };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(PhysicalDeviceIncompatibility::NoDepthStencilFormat(
                DEPTH_STENCIL_FORMAT,
            ));
        }

        // Choose surface format.
        let surface_formats = surface.get_formats(handle)?;
        // Choose one of R8G8B8A8_SRGB or B8G8R8A8_SRGB.
        let Some(surface_format) = surface_formats.into_iter().find(|format| {
            (format.format == vk::Format::R8G8B8A8_SRGB || format.format == vk::Format::B8G8R8A8_SRGB)
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) else {
            return Err(PhysicalDeviceIncompatibility::NoCompatibleSurfaceFormats(vec![
                vk::SurfaceFormatKHR::default()
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
                vk::SurfaceFormatKHR::default()
                    .format(vk::Format::B8G8R8A8_SRGB)
                    .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            ]));
        };

        // Choose surface presentation mode.
        let surface_present_modes = surface.get_present_modes(handle)?;
        // FIFO = waits for VSync, MAILBOX = waits for vsync, but will keep rendering (some frames will be thrown away, but reduces latency).
        const REQUESTED_PRESENTATION_MODE: vk::PresentModeKHR = vk::PresentModeKHR::MAILBOX;
        let presentation_mode = surface_present_modes
            .into_iter()
            .find(|&mode| mode == REQUESTED_PRESENTATION_MODE)
            // FIFO is always supported.
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // Calculate swapchain image count.
        let surface_capabilities = surface.get_capabilities(handle)?;
        let mut image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0 && image_count > surface_capabilities.max_image_count {
            image_count = surface_capabilities.max_image_count;
        }

        // Require compatible physical device properties.
        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
        unsafe { instance.get_physical_device_properties2(handle, &mut physical_device_properties) };
        let physical_device_properties = physical_device_properties.properties;

        if physical_device_properties.api_version < vulkan::MIN_VK_VERSION {
            return Err(PhysicalDeviceIncompatibility::IncompatibleVulkanVersion(
                physical_device_properties.api_version.into(),
            ));
        }

        let device_limits = physical_device_properties.limits;
        let supported_msaa_samples = device_limits.framebuffer_color_sample_counts
            & device_limits.framebuffer_depth_sample_counts
            & device_limits.framebuffer_stencil_sample_counts;
        let max_msaa_samples = [
            vk::SampleCountFlags::TYPE_1,
            vk::SampleCountFlags::TYPE_2,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_64,
        ]
        .into_iter()
        .rfind(|&sample_count| supported_msaa_samples.contains(sample_count))
        .unwrap_or(vk::SampleCountFlags::TYPE_1);

        // Require 128 bytes of push constant space.
        const REQUIRED_PUSH_CONSTANT_SIZE: u32 = 128;
        if device_limits.max_push_constants_size < REQUIRED_PUSH_CONSTANT_SIZE {
            return Err(PhysicalDeviceIncompatibility::NotEnoughPushConstantSpace(
                device_limits.max_push_constants_size,
                REQUIRED_PUSH_CONSTANT_SIZE,
            ));
        }

        let mut buffer_device_address_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut physical_device_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut buffer_device_address_features);
        unsafe { instance.get_physical_device_features2(handle, &mut physical_device_features) };

        // Require anisotropic filtering support.
        if physical_device_features.features.sampler_anisotropy == vk::FALSE {
            return Err(PhysicalDeviceIncompatibility::NoAnisotropicFiltering);
        }

        // Require buffer_device_address support.
        if buffer_device_address_features.buffer_device_address == vk::FALSE {
            return Err(PhysicalDeviceIncompatibility::NoBufferDeviceAddress);
        }

        Ok(PhysicalDevice {
            handle,
            primary_queue_family_idx,
            async_transfer_queue_family_idx,
            async_compute_queue_family_idx,
            is_discrete: physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            swapchain_format: surface_format,
            presentation_mode,
            depth_stencil_format: DEPTH_STENCIL_FORMAT,
            swapchain_image_count: image_count,
            supported_msaa_samples,
            max_msaa_samples,
            properties: physical_device_properties,
        })
    }

    pub fn get_unique_queue_families(&self) -> PropertyQueueList {
        let mut unique_queue_families = PropertyQueueList::new();
        unique_queue_families.push(self.primary_queue_family_idx);
        if let Some(async_transfer_queue_family_idx) = self.async_transfer_queue_family_idx {
            unique_queue_families.push(async_transfer_queue_family_idx);
        }
        if let Some(async_compute_queue_family_idx) = self.async_compute_queue_family_idx {
            unique_queue_families.push(async_compute_queue_family_idx);
        }
        unique_queue_families
    }
}
