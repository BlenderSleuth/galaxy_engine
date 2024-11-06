// Copyright (c) 2024 Ben Sutherland.

use ash::prelude::VkResult;
use ash::vk;

// Strip names in non-debug builds.
macro_rules! debug_only_name {
    ($format:literal$(,)? $($args: tt)*) => {
        &if cfg!(feature = "debug_info") { format!($format, $($args)*) } else { String::new() }
    };
    ($name:expr) => {
        if cfg!(feature = "debug_info") { $name } else { "".into() }
    };
}
pub(crate) use debug_only_name;

#[inline]
pub fn set_object_name<H: vk::Handle>(_device: &Device, _handle: H, _name: &str) -> VkResult<()> {
    set_object_name_with_ext(_device.extensions(), _handle, _name)
}

#[inline]
pub fn set_object_name_with_ext<H: vk::Handle>(_ext: &DeviceExtensions, _handle: H, _name: &str) -> VkResult<()> {
    #[cfg(feature = "debug_info")]
    {
        use std::ffi::CString;
        let name = CString::new(_name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(_handle)
            .object_name(&name);
        _ext.run_debug(|dbg| unsafe { dbg.set_debug_utils_object_name(&name_info) })
    }
    #[cfg(not(feature = "debug_info"))]
    {
        Ok(())
    }
}

#[cfg(feature = "debug_info")]
pub use debug_messenger::DebugMessenger;

use crate::vulkan::device::Device;
use crate::vulkan::extensions::DeviceExtensions;

#[cfg(feature = "debug_info")]
mod debug_messenger {
    use std::ffi::CStr;

    use ash::ext;

    use super::*;

    // These messages just clutter the log.
    const IGNORED_MESSAGE_IDS: &[u32] = &[
        0x20a0ac66, // BestPractices-NVIDIA-CreateDevice-PageableDeviceLocalMemory
        0x8b6f2f9a, // BestPractices-NVIDIA-AllocateMemory-SetPriority
        0xf00e92a8, // BestPractices-NVIDIA-CreateImage-Depth32Format
        // TODO: Re-enable this once combined image samplers can be used.
        0xf2b6a8e8, // BestPractices-NVIDIA-CreatePipelineLayout-SeparateSampler
        // TODO: Re-enable this when fences are pooled properly.
        0xa9f4ff68, // BestPractices-SyncObjects-HighNumberOfFences
    ];
    const IGNORED_MESSAGES: &[&CStr] =
        &[c"Validation Warning: [ WARNING-GPU-Assisted-Validation ] | MessageID = 0x24b5c69f | vkCreateDevice():  Internal Warning: Forcing VkPhysicalDeviceVulkan12Features::timelineSemaphore to VK_TRUE"];

    pub struct DebugMessenger {
        messenger: vk::DebugUtilsMessengerEXT,
        loader: ext::debug_utils::Instance,
    }

    impl DebugMessenger {
        pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> VkResult<Self> {
            let debug_utils_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
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

            if IGNORED_MESSAGE_IDS.contains(&(cd.message_id_number as u32)) {
                return vk::FALSE;
            }
            if IGNORED_MESSAGES.iter().any(|&m| Some(m) == cd.message_as_c_str()) {
                return vk::FALSE;
            }

            let message = unsafe { cd.message_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
            let message_id_name =
                unsafe { cd.message_id_name_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
            let message_id_number = cd.message_id_number;

            let _ = std::panic::catch_unwind(|| {
                log::log!(
                    level,
                    "{message_type:?} [{message_id_name} ({message_id_number})]\n\t{message}"
                );
            });

            vk::FALSE
        }
    }

    impl Drop for DebugMessenger {
        fn drop(&mut self) {
            unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) };
        }
    }
}
