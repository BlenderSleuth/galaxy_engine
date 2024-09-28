use std::ffi::{c_char, CStr};
use ash::{ext, khr, vk, Device, Entry, Instance};
use ash::prelude::VkResult;
use raw_window_handle::{DisplayHandle, WindowHandle};
use winit::dpi::PhysicalSize;
use crate::app::AppInfo;
use crate::{app, utils};

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum InitError {
    #[error("Library load failed: {0}")]
    LibraryLoadFailed(#[from] ash::LoadingError),
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error(
        "App requires Vulkan {}.{}.{} (Current: {}.{}.{}). Consider updating your graphics drivers",
        vk::api_version_major(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_minor(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_patch(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_major(*.0),
        vk::api_version_minor(*.0),
        vk::api_version_patch(*.0)
    )]
    IncompatibleVulkanVersion(u32),
    #[error("Instance extension error: {0}")]
    InstanceExtensionError(#[from] InstanceExtensionError),
    #[error("No compatible physical devices found.")]
    NoPhysicalDevices,
}

#[derive(thiserror::Error, Debug)]
pub enum InstanceExtensionError {
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Unable to find Vulkan extension: {0:?}")]
    ExtensionNotFound(&'static CStr),
}

struct DebugMessenger {
    messenger: vk::DebugUtilsMessengerEXT,
    loader: ext::debug_utils::Instance,
}

impl DebugMessenger {
    fn new(entry: &Entry, instance: &Instance) -> VkResult<Self> {
        let debug_utils_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
                    vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::debug_callback));

        let loader = ext::debug_utils::Instance::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_utils_ci, None) }?;

        Ok(Self { messenger, loader })
    }

    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        use std::borrow::Cow;

        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => log::Level::Warn,
        };

        if std::thread::panicking() {
            return vk::FALSE;
        }

        let cd = unsafe { *p_callback_data };

        let message_id_name =
            unsafe { cd.message_id_name_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
        let message = unsafe { cd.message_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
        let message_id_number = cd.message_id_number;

        let _ = std::panic::catch_unwind(|| {
            log::log!(level, "{message_type:?} [{message_id_name} (0x{message_id_number:x})]\n\t{message}");
        });

        vk::FALSE
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) };
    }
}

pub struct GalaxyEngine {
    entry: Entry,
    instance: Instance,
    debug_messenger: Option<DebugMessenger>,
    surface: vk::SurfaceKHR,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl GalaxyEngine {
    const MIN_VK_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
    const ENGINE_NAME: &'static CStr = c"Galaxy Engine";
    const ENGINE_VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");

    pub fn new(app_info: &AppInfo, display: DisplayHandle, window: WindowHandle, window_size: PhysicalSize<u32>) -> Result<Self, InitError> {
        // Setup Vulkan.
        let entry = unsafe { Entry::load() }?;

        // Check Vulkan API version.
        let api_version = unsafe { entry.try_enumerate_instance_version() }?.unwrap_or_else(|| vk::API_VERSION_1_0);

        // Require minimum VK version.
        if api_version < Self::MIN_VK_VERSION {
            return Err(InitError::IncompatibleVulkanVersion(api_version));
        }

        // Get instance extensions and layers
        let layers = Self::get_instance_layers(&entry, &app_info.flags)?;
        let extensions = Self::get_required_instance_extensions(&entry, &app_info.flags, display)?;

        let vk_app_info = vk::ApplicationInfo::default()
            .application_name(&app_info.name)
            .application_version(app_info.version)
            .engine_name(Self::ENGINE_NAME)
            .engine_version(utils::parse_version(Self::ENGINE_VERSION_STR))
            .api_version(api_version);

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&vk_app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(create_flags);

        let instance = unsafe { entry.create_instance(&instance_ci, None) }?;

        // Create debug messenger.
        let debug_messenger = if app_info.flags.contains(app::AppFlags::DEBUG) {
            Some(DebugMessenger::new(&entry, &instance)?)
        } else {
            None
        };

        // Create surface.
        let surface = unsafe { ash_window::create_surface(&entry, &instance, display.as_raw(), window.as_raw(), None) }?;
        let surface_fn = khr::surface::Instance::new(&entry, &instance);

        // Pick physical device.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err(InitError::NoPhysicalDevices);
        }

        // TODO: query specific swapchain properties.
        let required_device_extensions = vec![khr::swapchain::NAME];

        #[derive(Debug)]
        struct PhysicalDeviceProperties {
            physical_device: vk::PhysicalDevice,
            graphics_queue_family_idx: u32,
            present_queue_family_idx: u32,
            is_discrete: bool,
            swapchain_format: vk::SurfaceFormatKHR,
            presentation_mode: vk::PresentModeKHR,
            swap_extent: vk::Extent2D,
            image_count: u32,
            surface_capabilities: vk::SurfaceCapabilitiesKHR,
        }

        impl PhysicalDeviceProperties {
            const DEPTH_STENCIL_FORMAT: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;

            fn get_unique_queue_families(&self) -> Vec<u32> {
                let mut unique_queue_families = vec![self.graphics_queue_family_idx, self.present_queue_family_idx];
                unique_queue_families.sort_unstable();
                unique_queue_families.dedup();
                unique_queue_families
            }
        }

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
                let is_present_supported = unsafe { surface_fn.get_physical_device_surface_support(*physical_device, queue_family_idx, surface) }?;

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
            let surface_capabilities = unsafe { surface_fn.get_physical_device_surface_capabilities(*physical_device, surface) }?;
            let surface_formats = unsafe { surface_fn.get_physical_device_surface_formats(*physical_device, surface) }?;
            let surface_present_modes = unsafe { surface_fn.get_physical_device_surface_present_modes(*physical_device, surface) }?;

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

            // Choose swap extent.
            let swap_extent = if surface_capabilities.current_extent.width != u32::MAX {
                surface_capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: window_size.width.clamp(surface_capabilities.min_image_extent.width, surface_capabilities.max_image_extent.width),
                    height: window_size.height.clamp(surface_capabilities.min_image_extent.height, surface_capabilities.max_image_extent.height),
                }
            };

            let mut image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0 && image_count > surface_capabilities.max_image_count {
                image_count = surface_capabilities.max_image_count;
            }

            let device_properties = PhysicalDeviceProperties {
                physical_device: *physical_device,
                graphics_queue_family_idx: graphics_queue_family_idx.unwrap(),
                present_queue_family_idx: present_queue_family_idx.unwrap(),
                is_discrete: unsafe { instance.get_physical_device_properties(*physical_device).device_type } == vk::PhysicalDeviceType::DISCRETE_GPU,
                swapchain_format,
                presentation_mode,
                swap_extent,
                image_count,
                surface_capabilities,
            };

            // Prefer discrete GPU.
            if current_device_properties.is_none() || device_properties.is_discrete {
                current_device_properties = Some(device_properties);
            }
        }

        let Some(current_device_properties) = current_device_properties else {
            return Err(InitError::NoPhysicalDevices);
        };

        let unique_queue_families = current_device_properties.get_unique_queue_families();

        // Create logical device.
        let mut queue_cis = Vec::with_capacity(unique_queue_families.len());
        for unique_queue_family in unique_queue_families.iter() {
            queue_cis.push(vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*unique_queue_family)
                .queue_priorities(&[1.0]));
        }

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_extensions = utils::cstr_to_ptrs(required_device_extensions);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(current_device_properties.physical_device, &device_ci, None) }?;

        // Get queues.
        let graphics_queue = unsafe { device.get_device_queue(current_device_properties.graphics_queue_family_idx, 0) };
        let present_queue = unsafe { device.get_device_queue(current_device_properties.present_queue_family_idx, 0) };

        // Create swapchain.
        let (image_sharing_mode, queue_family_indices) = if unique_queue_families.len() > 1 {
            (vk::SharingMode::CONCURRENT, unique_queue_families)
        } else {
            (vk::SharingMode::EXCLUSIVE, Vec::new())
        };

        let swapchain_fn = khr::swapchain::Device::new(&instance, &device);
        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(current_device_properties.image_count)
            .image_format(current_device_properties.swapchain_format.format)
            .image_color_space(current_device_properties.swapchain_format.color_space)
            .image_extent(current_device_properties.swap_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(current_device_properties.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(current_device_properties.presentation_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_fn.create_swapchain(&swapchain_ci, None) }?;
        let swapchain_images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }?;

        let swapchain_image_views = swapchain_images.iter().map(|swapchain_image| {
            let image_view_ci = vk::ImageViewCreateInfo::default()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(current_device_properties.swapchain_format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { device.create_image_view(&image_view_ci, None) }
        }).collect::<VkResult<Vec<_>>>()?;

        Ok(Self { entry, instance, debug_messenger, surface, device, graphics_queue, present_queue, swapchain, swapchain_images, swapchain_image_views })
    }

    fn get_instance_layers(entry: &Entry, flags: &app::AppFlags) -> VkResult<Vec<*const c_char>> {
        // Query available layers.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;

        let mut required_layers = Vec::new();
        if flags.contains(app::AppFlags::DEBUG) {
            required_layers.push(c"VK_LAYER_KHRONOS_validation");
        }

        // Check all required layers are available. Not fatal if not found.
        required_layers.retain(|&required_layer| {
            if available_layers.iter().any(|&available_layer| {
                available_layer.layer_name_as_c_str() == Ok(required_layer)
            }) {
                true
            } else {
                log::warn!("Required layer not found: {:?}.", required_layer);
                false
            }
        });

        Ok(utils::cstr_to_ptrs(required_layers))
    }

    fn get_required_instance_extensions(entry: &Entry, flags: &app::AppFlags, display: DisplayHandle) -> Result<Vec<*const c_char>, InstanceExtensionError> {
        // Query available extensions
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }?;

        // Require platform windowing extensions. 
        // The returned extensions are pointers to static strings, so we can safely convert them back to CStr.
        let mut required_extensions = ash_window::enumerate_required_extensions(display.as_raw())?
            .iter()
            .map(|&ext| unsafe { CStr::from_ptr(ext) })
            .collect::<Vec<_>>();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME);
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME);
        }

        if flags.contains(app::AppFlags::DEBUG) {
            // Add debug messenger extension.
            required_extensions.push(ext::debug_utils::NAME);
        }

        // Check all required extensions are available.
        for required_extension in required_extensions.iter() {
            if !available_extensions.iter().any(|&available_extension| {
                available_extension.extension_name_as_c_str() == Ok(required_extension)
            }) {
                return Err(InstanceExtensionError::ExtensionNotFound(required_extension));
            }
        }

        Ok(utils::cstr_to_ptrs(required_extensions))
    }
}

impl Drop for GalaxyEngine {
    fn drop(&mut self) {
        // Drop image views.
        for image_view in self.swapchain_image_views.iter() {
            unsafe { self.device.destroy_image_view(*image_view, None) };
        }
        
        // Drop swapchain.
        let swapchain_fn = khr::swapchain::Device::new(&self.instance, &self.device);
        unsafe { swapchain_fn.destroy_swapchain(self.swapchain, None) };

        // Drop device.
        unsafe { self.device.destroy_device(None) };

        // Drop surface.
        let surface_fn = khr::surface::Instance::new(&self.entry, &self.instance);
        unsafe { surface_fn.destroy_surface(self.surface, None) };

        // Drop debug messenger.
        drop(self.debug_messenger.take());

        // Drop instance.
        unsafe { self.instance.destroy_instance(None) };
    }
}