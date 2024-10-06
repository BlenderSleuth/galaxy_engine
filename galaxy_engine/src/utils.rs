use std::ffi::{c_char, CStr};
use std::str::from_utf8;
use ash::vk;

pub(crate) const DEFAULT_SUBRESOURCE_RANGE: vk::ImageSubresourceRange =
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

pub(crate) fn get_aspect_for_format(format: vk::Format) -> vk::ImageAspectFlags {
    use vk::Format;
    match format {
        Format::D16_UNORM | Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub(crate) fn ktx_to_vulkan_format(format: ktx2::Format) -> vk::Format {
    use ktx2::Format;
    match format {
        Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_SRGB,
        _ => unimplemented!(),
    }
}

pub(crate) fn use_dedicated_allocation(dedicated_requirements: vk::MemoryDedicatedRequirements) -> bool {
    dedicated_requirements.requires_dedicated_allocation == vk::TRUE
        || dedicated_requirements.prefers_dedicated_allocation == vk::TRUE
}

pub(crate) fn cstr_to_ptrs(c_strs: &[&'static CStr]) -> Vec<*const c_char> {
    c_strs.iter().map(|cstr| cstr.as_ptr()).collect()
}

// From https://gist.github.com/rust-play/08f84ae7222deca0aaba4a5fd6b58278.
const fn subslice<T>(slice: &[T], range: std::ops::Range<usize>) -> &[T] {
    let mut slice = slice;
    let mut range = range;

    while range.start != 0 {
        slice = match slice {
            [_first, rest @ ..] => rest,
            _ => panic!("Index out of bounds"),
        };

        range.start -= 1;
        range.end -= 1;
    }

    loop {
        if slice.len() == range.end {
            return slice;
        }

        slice = match slice {
            [rest @ .., _last] => rest,
            _ => panic!("Index out of bounds"),
        }
    }
}

fn parse_num(bytes: &[u8]) -> u32 {
    match from_utf8(bytes) {
        // TODO: When `from_str_radix` is stable const, use it here (probable in the form of parse()).
        Ok(num) => match u32::from_str_radix(num, 10) {
            Ok(num) => num,
            Err(_) => panic!("Failed to parse number."),
        }
        Err(_) => panic!("Failed to convert to utf8."),
    }
}

pub(crate) fn parse_version(version: &str) -> u32 {
    let version = version.as_bytes();

    let mut idx_start = 0;
    let mut idx_curr = 0;

    while idx_curr < version.len() && version[idx_curr] != b'.' {
        idx_curr += 1;
    }
    let major = parse_num(subslice(version, idx_start..idx_curr));

    idx_curr += 1;
    idx_start = idx_curr;
    while idx_curr < version.len() && version[idx_curr] != b'.' {
        idx_curr += 1;
    }
    let minor = parse_num(subslice(version, idx_start..idx_curr));

    idx_curr += 1;
    idx_start = idx_curr;
    while idx_curr < version.len() && version[idx_curr] != b'.' {
        idx_curr += 1;
    }
    let patch = parse_num(subslice(version, idx_start..idx_curr));

    ash::vk::make_api_version(0, major, minor, patch)
}
