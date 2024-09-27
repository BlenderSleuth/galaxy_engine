use std::ffi::CString;

use bitflags::bitflags;

bitflags! {
    pub struct AppFlags: u32 {
        // #[cfg(debug_assertions)]
        const DEBUG = 1 << 0;
    }
}

pub struct AppInfo {
    pub name: CString,
    pub version: u32,
    pub flags: AppFlags,
}

impl AppInfo {
    pub fn new(name: &str, version: u32, flags: AppFlags) -> AppInfo {
        AppInfo {
            name: CString::new(name).unwrap_or(c"Unknown".into()),
            version,
            flags,
        }
    }
}