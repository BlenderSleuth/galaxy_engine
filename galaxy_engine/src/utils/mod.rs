// Copyright (c) 2024 Ben Sutherland.

mod arc_final_owner;
pub use arc_final_owner::ArcFinalOwner;
mod formats;
pub use formats::*;
mod array;
mod config;
mod extensions;

use std::ffi::{c_char, CStr};
use std::ops::{Add, Sub};

pub use array::*;
pub use config::*;
pub use extensions::*;

pub(crate) const fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

pub(crate) fn cstr_to_ptrs(c_strs: &[&'static CStr]) -> Vec<*const c_char> {
    c_strs.iter().map(|cstr| cstr.as_ptr()).collect()
}

pub(crate) const fn parse_num(num: &'static str) -> u32 {
    match u32::from_str_radix(num, 10) {
        Ok(num) => num,
        Err(_) => panic!("Failed to parse number."),
    }
}

macro_rules! parse_version {
    ($version:expr) => {{
        use crate::utils::parse_num;
        let [major, minor, patch] = const_format::str_split!($version, '.');
        ash::vk::make_api_version(0, parse_num(major), parse_num(minor), parse_num(patch))
    }};
    () => {};
}
pub(crate) use parse_version;

//pub(crate) const fn parse_version(version: &'static str) -> u32 {
//    let [major, minor, patch] = const_format::str_split!(version, '.');
//    vk::make_api_version(0, parse_num(major), parse_num(minor), parse_num(patch))
//}
