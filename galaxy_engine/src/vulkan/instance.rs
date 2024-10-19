// Copyright (c) 2024. Ben Sutherland

use std::ffi::CStr;

use ash::vk;
use raw_window_handle::DisplayHandle;

use crate::app::AppInfo;
use crate::vulkan::debug::DebugMessenger;
use crate::{app, utils, vulkan};

#[derive(Debug, thiserror::Error)]
pub enum InstanceInitError {
    #[error("Library load failed: {0}")]
    LibraryLoadFailed(#[from] ash::LoadingError),
    #[error("Vulkan call failed: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Incompatible Vulkan version: {0}")]
    IncompatibleVulkanVersion(vulkan::IncompatibleVulkanVersion),
    #[error("Unable to find required Vulkan instance extension: {0:?}")]
    ExtensionNotFound(&'static CStr),
}

pub struct Instance {
    #[cfg(feature = "debug_info")]
    debug_messenger: Option<DebugMessenger>,
    loader: ash::Instance,
    entry: ash::Entry,
}

impl Instance {
    pub fn new(app_info: &AppInfo, display: DisplayHandle) -> Result<Self, InstanceInitError> {
        // Setup Vulkan.
        let entry = unsafe { ash::Entry::load() }?;

        // Check Vulkan API version.
        let api_version = unsafe { entry.try_enumerate_instance_version() }?.unwrap_or_else(|| vk::API_VERSION_1_0);

        // Require minimum VK version.
        if api_version < super::MIN_VK_VERSION {
            return Err(InstanceInitError::IncompatibleVulkanVersion(api_version.into()));
        }

        // Instance layers.
        let layers = {
            // Query available instance layers.
            let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;

            let mut optional_layers = Vec::new();
            #[cfg(feature = "debug_info")]
            if app_info.flags.contains(app::AppFlags::DEBUG) {
                optional_layers.push(c"VK_LAYER_KHRONOS_validation");
            }

            // Check which optional layers are available. Not fatal if not found.
            optional_layers.retain(|&optional_layer| {
                if available_layers
                    .iter()
                    .any(|&available_layer| available_layer.layer_name_as_c_str() == Ok(optional_layer))
                {
                    true
                } else {
                    log::warn!("Requested optional layer not found: {optional_layer:?}.");
                    false
                }
            });
            utils::cstr_to_ptrs(&optional_layers)
        };

        // Instance extensions.
        #[cfg(feature = "debug_info")]
        let mut enabled_debug_utils = false;
        let instance_extensions = {
            // Query available extensions.
            let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }?;
            let mut required_extensions = Vec::new();

            // Require platform windowing extensions.
            // The returned extensions are pointers to static strings, so we can safely convert them back to CStr.
            required_extensions.extend(
                ash_window::enumerate_required_extensions(display.as_raw())?
                    .iter()
                    .map(|&ext| unsafe { CStr::from_ptr::<'static>(ext) }),
            );

            // MacOS compatibility.
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                required_extensions.push(ash::khr::portability_enumeration::NAME);
            }

            // Check all required extensions are available.
            for required_extension in required_extensions.iter() {
                if !available_extensions
                    .iter()
                    .any(|&available_extension| available_extension.extension_name_as_c_str() == Ok(required_extension))
                {
                    return Err(InstanceInitError::ExtensionNotFound(required_extension));
                }
            }

            #[cfg(feature = "debug_info")]
            if app_info.flags.contains(app::AppFlags::DEBUG) {
                // Add debug messenger extension if available.
                if available_extensions.iter().any(|&available_extension| {
                    available_extension.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME)
                }) {
                    required_extensions.push(ash::ext::debug_utils::NAME);
                    enabled_debug_utils = true;
                } else {
                    log::warn!("Debug flag enabled, but debug utils instance extension not available.");
                }
            }

            utils::cstr_to_ptrs(&required_extensions)
        };

        let vk_app_info = vk::ApplicationInfo::default()
            .application_name(&app_info.name)
            .application_version(app_info.version)
            .engine_name(super::ENGINE_NAME)
            .engine_version(utils::parse_version(super::ENGINE_VERSION_STR))
            .api_version(super::MIN_VK_VERSION);

        // MacOS compatibility.
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&vk_app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&instance_extensions)
            .flags(create_flags);

        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        // Create debug messenger.
        #[cfg(feature = "debug_info")]
        let debug_messenger = if enabled_debug_utils {
            Some(DebugMessenger::new(&entry, &instance)?)
        } else {
            None
        };

        Ok(Self {
            debug_messenger,
            loader: instance,
            entry,
        })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn loader(&self) -> &ash::Instance {
        &self.loader
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // Drop debug messenger.
        #[cfg(feature = "debug_info")]
        {
            self.debug_messenger = None;
        }

        // Drop instance.
        unsafe { self.loader.destroy_instance(None) };

        // Entry is automatically dropped.
    }
}
