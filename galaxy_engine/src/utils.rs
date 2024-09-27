use std::ffi::{c_char, CStr};
use std::str::from_utf8;

pub(crate) fn cstr_to_ptrs(c_strs: Vec<&'static CStr>) -> Vec<*const c_char> {
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
