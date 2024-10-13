use std::ffi::{c_char, CStr};
use std::mem::ManuallyDrop;
use std::str::from_utf8;
use std::sync::Arc;
use ash::vk;

pub fn viewport_extent(viewport: vk::Viewport) -> vk::Extent2D {
    vk::Extent2D {
        width: viewport.width as u32,
        height: viewport.height as u32,
    }
}

struct AssertLessThanOrEqual<const N1: usize, const N2: usize>;
impl<const N1: usize, const N2: usize> AssertLessThanOrEqual<N1, N2> {
    const OK: () = assert!(N1 <= N2, "N1 must be <= N2.");
}

pub(crate) fn arrayvec_from_array<T, const N1: usize, const N2: usize>(array: [T; N1]) -> arrayvec::ArrayVec<T, N2> {
    let _ = AssertLessThanOrEqual::<N1, N2>::OK;
    array.into_iter().collect()
}

// Strip names in release builds.
macro_rules! debug_only_name {
    ($format:literal$(,)? $($args: tt)*) => {
        &if cfg!(feature = "debug_info") { format!($format, $($args)*) } else { String::new() }
    };
    ($name:expr) => {
        if cfg!(feature = "debug_info") { $name } else { "".into() }
    };
}
pub(crate) use debug_only_name;

// Used to allow sharing of objects, but also ensuring that it is destroyed at the appropriate time.
pub struct ArcFinalOwner<T>(ManuallyDrop<Arc<T>>);

#[derive(Debug)]
pub enum FinalOwnerError {
    NotLastOwner,
}

impl<T> ArcFinalOwner<T> {
    pub fn new(value: T) -> Self {
        Self(ManuallyDrop::new(Arc::new(value)))
    }

    pub unsafe fn destroy_as_final(&mut self, destroy: impl FnOnce(&mut T)) -> Result<(), FinalOwnerError> {
        // Get shared item and drop it. Ensure we are the last owner of the shared reference.
        let object = unsafe { ManuallyDrop::take(&mut self.0) };
        match Arc::try_unwrap(object) {
            Ok(mut object) => {
                destroy(&mut object);
                Ok(())
            },
            Err(arc) => {
                log::error!("Not last owner of Vulkan object.");
                self.0 = ManuallyDrop::new(arc);
                Err(FinalOwnerError::NotLastOwner)
            }
        }
    }
}

impl<T> std::ops::Deref for ArcFinalOwner<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
