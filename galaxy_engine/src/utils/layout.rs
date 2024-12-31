// Copyright (c) 2024 Ben Sutherland.

pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}
