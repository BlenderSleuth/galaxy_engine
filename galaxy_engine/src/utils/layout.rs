// Copyright (c) 2024 Ben Sutherland.

use std::alloc::{Layout, LayoutError};
use std::ops::Range;

pub struct CStructLayout {
    pub layout: Layout,
    pub ranges: Vec<Range<usize>>,
}

impl CStructLayout {
    // Modified from Layout::extend() docs.
    pub fn new(fields: impl Iterator<Item = Layout>) -> Result<Self, LayoutError> {
        let mut layout = Layout::from_size_align(0, 1)?;

        // Calculate offset and size for each field.
        let offset_and_size: Vec<(usize, usize)> = fields
            .map(|field| {
                let (new_layout, offset) = layout.extend(field)?;
                layout = new_layout;
                Ok((offset, field.size()))
            })
            .collect::<Result<_, LayoutError>>()?;
        let layout = layout.pad_to_align();

        // Calculate field ranges.
        let ranges = offset_and_size
            .into_iter()
            .map(|(offset, size)| offset..offset + size)
            .collect();

        Ok(Self { layout, ranges })
    }
    //pub fn iter_fields<'a>(&self, buffer: &'a mut [u8]) -> impl Iterator<Item = &'a mut [u8]> + use<'a, '_> {
    //    self.ranges.iter().cloned().map(|range| &mut buffer[range])
    //}
    //pub fn get_field_memory<'a, T: bytemuck::Pod>(&self, buffer: &'a mut [u8], index: usize) -> &'a mut T {
    //    bytemuck::from_bytes_mut(&mut buffer[self.ranges[index].clone()])
    //}
}

// Stable polyfill for unstable layout functions.
pub trait LayoutExt {
    fn padding_needed_for_pf(&self, align: usize) -> usize;
    fn repeat_pf(&self, n: usize) -> Result<(Layout, usize), LayoutError>;
}

impl LayoutExt for Layout {
    //#[unstable(feature = "alloc_layout_extra", issue = "55724")]
    //#[rustc_const_unstable(feature = "const_alloc_layout", issue = "67521")]
    //#[must_use = "this returns the padding needed, \
    //              without modifying the `Layout`"]
    #[inline]
    fn padding_needed_for_pf(&self, align: usize) -> usize {
        let len = self.size();

        // Rounded up value is:
        //   len_rounded_up = (len + align - 1) & !(align - 1);
        // and then we return the padding difference: `len_rounded_up - len`.
        //
        // We use modular arithmetic throughout:
        //
        // 1. align is guaranteed to be > 0, so align - 1 is always
        //    valid.
        //
        // 2. `len + align - 1` can overflow by at most `align - 1`,
        //    so the &-mask with `!(align - 1)` will ensure that in the
        //    case of overflow, `len_rounded_up` will itself be 0.
        //    Thus the returned padding, when added to `len`, yields 0,
        //    which trivially satisfies the alignment `align`.
        //
        // (Of course, attempts to allocate blocks of memory whose
        // size and padding overflow in the above manner should cause
        // the allocator to yield an error anyway.)

        let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
        len_rounded_up.wrapping_sub(len)
    }

    //#[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    fn repeat_pf(&self, n: usize) -> Result<(Layout, usize), LayoutError> {
        // This cannot overflow. Quoting from the invariant of Layout:
        // > `size`, when rounded up to the nearest multiple of `align`,
        // > must not overflow isize (i.e., the rounded value must be
        // > less than or equal to `isize::MAX`)
        let padded_size = self.size() + self.padding_needed_for_pf(self.align());
        let alloc_size = padded_size.checked_mul(n).unwrap();

        // The safe constructor is called here to enforce the isize size limit.
        let layout = Layout::from_size_align(alloc_size, self.align())?;
        Ok((layout, padded_size))
    }
}

//pub(crate) const fn align_up(value: u32, alignment: u32) -> u32 {
//    (value + alignment - 1) & !(alignment - 1)
//}
