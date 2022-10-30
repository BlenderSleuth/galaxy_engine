use std::ffi;

/// Copied from wgpu-hal
///
/// Construct a `CStr` from a byte slice, up to the first zero byte.
///
/// Return a `CStr` extending from the start of `bytes` up to and
/// including the first zero byte. If there is no zero byte in
/// `bytes`, return `None`.
///
/// This can be removed when `CStr::from_bytes_until_nul` is stabilized.
/// ([#95027](https://github.com/rust-lang/rust/issues/95027))
pub(crate) fn cstr_from_bytes_until_nul(bytes: &[ffi::c_char]) -> Option<&ffi::CStr> {
    if bytes.contains(&0) {
        // Safety for `CStr::from_ptr`:
        // - We've ensured that the slice does contain a null terminator.
        // - The range is valid to read, because the slice covers it.
        // - The memory won't be changed, because the slice borrows it.
        unsafe { Some(ffi::CStr::from_ptr(bytes.as_ptr())) }
    } else {
        None
    }
}
