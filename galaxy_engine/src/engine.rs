use std::ffi::{c_char, CStr};
use ash::{ext, khr, vk, Entry, Instance};
use ash::prelude::VkResult;
use raw_window_handle::{DisplayHandle, WindowHandle};
use winit::dpi::PhysicalSize;

use crate::{app, utils, device, swapchain};
use app::AppInfo;
use device::Device;
use swapchain::Swapchain;

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum EngineInitError {
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
    #[error("Device init error: {0}")]
    DeviceInitError(#[from] device::DeviceInitError),
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
    
    pub unsafe fn destroy(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) };
    }
}

pub struct GalaxyEngine {
    entry: Entry,
    instance: Instance,
    debug_messenger: Option<DebugMessenger>,
    surface: vk::SurfaceKHR,
    device: Device,
    swapchain: Swapchain,
}

impl GalaxyEngine {
    const MIN_VK_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
    const ENGINE_NAME: &'static CStr = c"Galaxy Engine";
    const ENGINE_VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");

    pub fn new(app_info: &AppInfo, display: DisplayHandle, window: WindowHandle, window_size: PhysicalSize<u32>) -> Result<Self, EngineInitError> {
        // Setup Vulkan.
        let entry = unsafe { Entry::load() }?;

        // Check Vulkan API version.
        let api_version = unsafe { entry.try_enumerate_instance_version() }?.unwrap_or_else(|| vk::API_VERSION_1_0);

        // Require minimum VK version.
        if api_version < Self::MIN_VK_VERSION {
            return Err(EngineInitError::IncompatibleVulkanVersion(api_version));
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

        // Create device.
        let device = Device::new(&entry, &instance, surface, window_size)?;

        // Create swapchain.
        let swapchain = Swapchain::new(&instance, &device, surface, None)?;
        
        
        

        Ok(Self { entry, instance, debug_messenger, surface, device, swapchain })
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
        // Drop swapchain.
        unsafe { self.swapchain.destroy(&self.device) };

        // Drop device.
        unsafe { self.device.destroy() };

        // Drop surface.
        let surface_fn = khr::surface::Instance::new(&self.entry, &self.instance);
        unsafe { surface_fn.destroy_surface(self.surface, None) };

        // Drop debug messenger.
        if let Some(debug_messenger) = &mut self.debug_messenger {
            unsafe { debug_messenger.destroy() };
        }

        // Drop instance.
        unsafe { self.instance.destroy_instance(None) };
    }
}