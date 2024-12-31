// Copyright (c) 2024 Ben Sutherland.

mod arc_final_owner;

pub use arc_final_owner::ArcFinalOwner;
mod formats;
pub use formats::*;
mod array;
mod config;
pub(crate) use config::*;
mod extensions;
pub use extensions::*;
mod layout;
mod scope_guard;
use std::ffi::{c_char, CStr};

use ash::vk;
pub use layout::align_up;
pub use scope_guard::ScopeGuard;

pub(crate) fn cstr_to_ptrs(c_strs: &[&'static CStr]) -> Vec<*const c_char> {
    c_strs.iter().map(|cstr| cstr.as_ptr()).collect()
}

pub(crate) const fn parse_num(num: &'static str) -> u32 {
    match u32::from_str_radix(num, 10) {
        Ok(num) => num,
        Err(_) => panic!("Failed to parse number."),
    }
}

//macro_rules! parse_version {
//    ($version:expr) => {{
//        use crate::utils::parse_num;
//        let [major, minor, patch] = const_format::str_split!($version, '.');
//        ash::vk::make_api_version(0, parse_num(major), parse_num(minor), parse_num(patch))
//    }};
//    () => {};
//}
//pub(crate) use parse_version;

pub const fn pkg_version() -> u32 {
    vk::make_api_version(
        0,
        parse_num(env!("CARGO_PKG_VERSION_MAJOR")),
        parse_num(env!("CARGO_PKG_VERSION_MINOR")),
        parse_num(env!("CARGO_PKG_VERSION_PATCH")),
    )
}
