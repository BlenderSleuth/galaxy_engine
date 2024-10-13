use crate::device::{Device, LoadedExtensions};
use ash::prelude::VkResult;
use ash::vk;

#[inline]
pub fn set_object_name<H: vk::Handle>(_device: &Device, _handle: H, _name: &str) -> VkResult<()> {
    set_object_name_with_ext(_device.ext(), _handle, _name)
}

#[inline]
pub fn set_object_name_with_ext<H: vk::Handle>(_ext: &LoadedExtensions, _handle: H, _name: &str) -> VkResult<()> {
    #[cfg(feature = "debug_info")]
    {
        use std::ffi::CString;
        let name = CString::new(_name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(_handle)
            .object_name(&name);
        unsafe { _ext.run_debug(|dbg| dbg.set_debug_utils_object_name(&name_info)) }
    }
    #[cfg(not(feature = "debug_info"))]
    {
        Ok(())
    }
}

#[cfg(feature = "debug_info")]
pub use debug_messenger::DebugMessenger;

#[cfg(feature = "debug_info")]
mod debug_messenger {
    use super::*;
    use std::ffi::CStr;
    use ash::ext;

    pub struct DebugMessenger {
        messenger: vk::DebugUtilsMessengerEXT,
        loader: ext::debug_utils::Instance,
    }

    impl DebugMessenger {
        pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> VkResult<Self> {
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
}