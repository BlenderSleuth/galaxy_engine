use std::error::Error;
use std::ffi::{c_char, CStr, CString};
use std::fmt;

use ash::{
    extensions::{ext, khr},
    vk, Device, Entry, Instance,
};
use ash_window;
use cstr::cstr;
use log;
use raw_window_handle::RawDisplayHandle;
use thiserror::Error;

use crate::app;
use crate::constants::*;
use crate::utils;

#[derive(Clone, PartialEq, Eq, Debug, Error)]
#[non_exhaustive]
pub enum InitError {
    // Stores CStr NulError
    #[error("provided app name has an internal null character")]
    AppNameHasInternalNull(#[source] std::ffi::NulError),
    // Stores VkResult from function call
    #[error("vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    // Stores the incompatible vulkan driver API version
    #[error(
        "App requires Vulkan 1.2. Vulkan driver API version is incompatible: {}.{}.{}",
        vk::api_version_major(0),
        vk::api_version_minor(0),
        vk::api_version_patch(0)
    )]
    VulkanIncompatibleVersion(u32),
}

pub struct Renderer {
    instance: Instance,
    entry: Entry,
}

impl Renderer {
    pub fn new(app_info: &app::AppInfo, display: RawDisplayHandle) -> Result<Self, InitError> {
        // Setup Vulkan
        let entry = Entry::linked();

        // Check Vulkan API versions
        let driver_api_version = match entry.try_enumerate_instance_version() {
            Ok(Some(version)) => version,
            Ok(None) => vk::API_VERSION_1_0,
            Err(err) => {
                log::warn!("Vulkan instance version query failed: {:?}", err);
                return Err(InitError::VulkanError(err));
            }
        };

        // Require minimum VK version
        if vk::api_version_major(driver_api_version) <= vk::api_version_major(MIN_VK_VERSION)
            && vk::api_version_minor(driver_api_version) < vk::api_version_major(MIN_VK_VERSION)
        {
            return Err(InitError::VulkanIncompatibleVersion(driver_api_version));
        }

        // Convert app name to a c-string
        let app_name =
            CString::new(app_info.name).map_err(|e| InitError::AppNameHasInternalNull(e))?;
        let app_info_vk = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(app_info.version)
            .engine_name(ENGINE_NAME_C)
            .engine_version(ENGINE_VERSION)
            .api_version(MIN_VK_VERSION);

        // Get instance extensions and layers
        let layers = Self::get_instance_layers(&entry, &app_info.flags)?;
        let extensions = Self::get_required_instance_extensions(&entry, &app_info.flags, display)?;

        let instance_ci = vk::InstanceCreateInfo::builder()
            .application_info(&app_info_vk)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        let instance = unsafe {
            entry.create_instance(&instance_ci, None).map_err(|e| {
                log::error!("Vulkan create_instance failed");
                InitError::VulkanError(e)
            })?
        };

        Ok(Self { instance, entry })
    }

    fn get_instance_layers(
        entry: &Entry,
        flags: &app::AppFlags,
    ) -> Result<Vec<*const c_char>, InitError> {
        // Query available layers
        let available_layers = entry.enumerate_instance_layer_properties().map_err(|e| {
            log::error!("Vulkan enumerate_instance_layer_properties failed");
            InitError::VulkanError(e)
        })?;

        const VALIDATION_LAYER: &CStr = cstr!("VK_LAYER_KHRONOS_validation");

        let mut required_layers: Vec<&CStr> = Vec::new();

        if flags.contains(app::AppFlags::DEBUG) {
            required_layers.push(VALIDATION_LAYER);
        }

        // Check required against available layers
        required_layers.retain(|&layer| {
            if available_layers.iter().any(|&available_layer| {
                utils::cstr_from_bytes_until_nul(&available_layer.layer_name) == Some(layer)
            }) {
                true
            } else {
                log::warn!("Unable to find Vulkan layer: {}", layer.to_string_lossy());
                false
            }
        });

        // Convert to pointers
        let required_layers_raw = required_layers
            .iter()
            .map(|&layer| layer.as_ptr())
            .collect();

        Ok(required_layers_raw)
    }

    fn get_required_instance_extensions(
        entry: &Entry,
        flags: &app::AppFlags,
        display: RawDisplayHandle,
    ) -> Result<Vec<*const c_char>, InitError> {
        // Query available extensions
        let available_extensions = entry
            .enumerate_instance_extension_properties(None)
            .map_err(|e| {
                log::error!("Vulkan enumerate_instance_extension_properties failed");
                InitError::VulkanError(e)
            })?;

        // Require platform windowing extensions
        let mut required_extensions: Vec<&'static CStr> =
            ash_window::enumerate_required_extensions(display)?
                .iter()
                .map(|ext| unsafe { CStr::from_ptr(*ext) })
                .collect();

        // TODO: Device extensions
        // Require VK_KHR_synchronization2
        // required_extensions.push(khr::Synchronization2::name());

        // // Require raytracing Extensions
        // if flags.contains(app::AppFlags::RAYTRACING) {
        //     required_extensions.push(khr::AccelerationStructure::name());
        //     required_extensions.push(khr::RayTracingPipeline::name());
        //     //required_extensions.push(khr::));
        // }
        if flags.contains(app::AppFlags::DEBUG) {
            // Add debug messenger extension
            required_extensions.push(ext::DebugUtils::name());
        }

        // Check required against available extensions
        required_extensions.retain(|&ext| {
            if available_extensions.iter().any(|&available_ext| {
                utils::cstr_from_bytes_until_nul(&available_ext.extension_name) == Some(ext)
            }) {
                true
            } else {
                log::warn!("Unable to find Vulkan extension: {}", ext.to_string_lossy());
                false
            }
        });

        // Convert to pointers
        let equired_extensions_raw = required_extensions
            .iter()
            .map(|&ext| ext.as_ptr())
            .collect();

        Ok(equired_extensions_raw)
    }

    pub fn main_loop(&self) {}
}
