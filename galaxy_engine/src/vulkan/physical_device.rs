// Copyright (c) 2024 Ben Sutherland.

use std::ffi::CStr;
use std::num::NonZeroU32;

use arrayvec::ArrayVec;
use ash::vk;

use crate::vulkan;
use crate::vulkan::surface::Surface;

// Memory type that can be used for buffers that are written to every frame (all of Host Visible, Host Coherent and Device Local).
#[derive(Copy, Clone, Debug)]
pub struct VolatileMemoryType {
    pub type_bits: NonZeroU32,
    pub size: vk::DeviceSize,
}

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
    #[error("{0} not supported")]
    FeatureNotSupported(&'static str),
    #[error("Uniform memory not available")]
    NoUniformMemoryAvailable,
}

pub struct PhysicalDeviceProperties {
    pub base: vk::PhysicalDeviceProperties,
    pub descriptor_buffer: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>,
}

impl PhysicalDeviceProperties {
    pub fn new(instance: &ash::Instance, handle: vk::PhysicalDevice) -> PhysicalDeviceProperties {
        let mut descriptor_buffer_properties = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();

        // Require compatible physical device properties.
        let mut physical_device_properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut descriptor_buffer_properties);
        unsafe { instance.get_physical_device_properties2(handle, &mut physical_device_properties) };

        PhysicalDeviceProperties {
            base: physical_device_properties.properties,
            descriptor_buffer: descriptor_buffer_properties,
        }
    }
}

#[derive(Default)]
pub struct PhysicalDeviceFeatures {
    pub features: vk::PhysicalDeviceFeatures,
    pub features11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub features12: vk::PhysicalDeviceVulkan12Features<'static>,
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
    pub volatile_memory_type: VolatileMemoryType,
    pub mem_properties: vk::PhysicalDeviceMemoryProperties,
    pub properties: PhysicalDeviceProperties,
    pub enabled_features: PhysicalDeviceFeatures,
}

//noinspection RsUnresolvedPath
pub type PropertyQueueList<T = u32> = ArrayVec<T, { PhysicalDevice::MAX_QUEUE_FAMILIES }>;
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
        const REQUESTED_PRESENTATION_MODE: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;
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

        let physical_device_properties = PhysicalDeviceProperties::new(instance, handle);

        let device_properties = &physical_device_properties.base;
        if device_properties.api_version < vulkan::MIN_VK_VERSION {
            return Err(PhysicalDeviceIncompatibility::IncompatibleVulkanVersion(
                device_properties.api_version.into(),
            ));
        }

        let device_limits = device_properties.limits;
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

        let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features11)
            .push_next(&mut features12);
        unsafe { instance.get_physical_device_features2(handle, &mut physical_device_features) };
        let features = physical_device_features.features;

        let mut enabled_features = PhysicalDeviceFeatures::default();

        macro_rules! check_and_enable_feature {
            ($group:ident.$feature:ident) => {
                if $group.$feature == vk::FALSE {
                    return Err(PhysicalDeviceIncompatibility::FeatureNotSupported(stringify!(
                        $feature
                    )));
                }
                enabled_features.$group.$feature = vk::TRUE;
            };
        }

        // Require anisotropic filtering support.
        check_and_enable_feature!(features.sampler_anisotropy);

        // Require shader draw parameters support.
        check_and_enable_feature!(features11.shader_draw_parameters);

        // Validation layers require some features enabled.
        if cfg!(feature = "debug_info") {
            check_and_enable_feature!(features.fragment_stores_and_atomics);
            check_and_enable_feature!(features.vertex_pipeline_stores_and_atomics);
            check_and_enable_feature!(features12.uniform_and_storage_buffer8_bit_access);
            check_and_enable_feature!(features12.timeline_semaphore);
        }

        // Require runtime descriptor array support.
        check_and_enable_feature!(features12.runtime_descriptor_array);

        // Require buffer_device_address support.
        check_and_enable_feature!(features12.buffer_device_address);

        // Require scalar block layout support. TODO: Temporary solution until https://github.com/shader-slang/slang/issues/5806 gets resolved.
        check_and_enable_feature!(features12.scalar_block_layout);

        // Require descriptor indexing support.
        //check_and_enable_feature!(features12.descriptor_indexing);
        // Require draw indirect count.
        //check_and_enable_feature!(features12.draw_indirect_count);

        let mem_properties = unsafe { instance.get_physical_device_memory_properties(handle) };

        // Choose the memory type to use for uniforms updated every frame.
        //TODO: handle when there are no volatile memory types available.
        let mut volatile_memory_types: Vec<_> = mem_properties
            .memory_types
            .iter()
            .take(mem_properties.memory_type_count as usize)
            .enumerate()
            .filter_map(|(idx, mem_type)| {
                if mem_type.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ) {
                    Some(VolatileMemoryType {
                        type_bits: NonZeroU32::new(1 << idx as u32).unwrap(),
                        size: mem_properties.memory_heaps[mem_type.heap_index as usize].size,
                    })
                } else {
                    None
                }
            })
            .collect();
        let volatile_memory_type_with_max_size = volatile_memory_types
            .iter()
            .max_by(|a, b| a.size.cmp(&b.size))
            .copied()
            .ok_or(PhysicalDeviceIncompatibility::NoUniformMemoryAvailable)?;
        volatile_memory_types.retain(|mem| mem.size == volatile_memory_type_with_max_size.size);
        // Combine type bits.
        let volatile_memory_type = volatile_memory_types
            .iter()
            .fold(volatile_memory_type_with_max_size, |acc, mem| VolatileMemoryType {
                type_bits: NonZeroU32::new(acc.type_bits.get() | mem.type_bits.get()).unwrap(),
                size: acc.size,
            });

        Ok(PhysicalDevice {
            handle,
            primary_queue_family_idx,
            async_transfer_queue_family_idx,
            async_compute_queue_family_idx,
            is_discrete: device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            swapchain_format: surface_format,
            presentation_mode,
            depth_stencil_format: DEPTH_STENCIL_FORMAT,
            swapchain_image_count: image_count,
            supported_msaa_samples,
            max_msaa_samples,
            volatile_memory_type,
            mem_properties,
            properties: physical_device_properties,
            enabled_features,
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
