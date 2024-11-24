// Copyright (c) 2024 Ben Sutherland.

//struct AssertLessThanOrEqual<const N1: usize, const N2: usize>;
//impl<const N1: usize, const N2: usize> AssertLessThanOrEqual<N1, N2> {
//    const OK: () = assert!(N1 <= N2, "N1 must be <= N2.");
//}
//
//pub fn arrayvec_from_array<T, const N1: usize, const N2: usize>(array: [T; N1]) -> ArrayVec<T, N2> {
//    let _ = AssertLessThanOrEqual::<N1, N2>::OK;
//    array.into_iter().collect()
//}
