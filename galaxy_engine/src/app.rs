use bitflags::bitflags;

bitflags! {
    pub struct AppFlags: u32 {
        const DEBUG = 1 << 0;
        const RAYTRACING = 1 << 1;
    }
}

pub struct AppInfo<'a> {
    pub name: &'a str,
    pub version: u32,
    pub flags: AppFlags,
}
