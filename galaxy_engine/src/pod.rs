// Copyright (c) 2024-2025 Ben Sutherland.

// Copies of vk structs to mark them as Pod.

pub(crate) mod vk {
    use bytemuck::{Pod, Zeroable};

    // Copied from vk::DrawIndexedIndirectCommand.
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct DrawIndexedIndirectCommand {
        pub index_count: u32,
        pub instance_count: u32,
        pub first_index: u32,
        pub vertex_offset: i32,
        pub first_instance: u32,
    }
}
