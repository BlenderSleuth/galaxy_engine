use cstr::cstr;
use std::ffi::CStr;

use ash::vk::API_VERSION_1_2;

pub const ENGINE_NAME: &str = "Galaxy Engine";
pub const ENGINE_NAME_C: &CStr = cstr!("Galaxy Engine");
pub const ENGINE_VERSION: u32 = 1;
pub const MIN_VK_VERSION: u32 = API_VERSION_1_2;
