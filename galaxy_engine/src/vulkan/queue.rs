// Copyright (c) 2024. Ben Sutherland

use std::marker::PhantomData;

use ash::vk;

pub mod queue_type {
    // We need a graphics queue for rendering, and we also get a "sync" compute queue for free.
    // From the spec: "If an implementation exposes any queue family that supports graphics operations,
    // at least one queue family of at least one physical vulkan exposed by the implementation must
    // support both graphics and compute operations."

    // Additionally from the spec: "All commands that are allowed on a queue that supports transfer
    // operations are also allowed on a queue that supports either graphics or compute operations."
    // So we can use the graphics queue for "sync" transfer operations as well.

    // Not from the spec, but on good authority, "No such hardware exists" that supports graphics and
    // presentation without a graphics and present queue. Thus, this primary queue is also used for presentation.
    // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th

    pub trait QueueType {}
    // This is the primary, mandatory queue, with support for graphics, compute, transfer and present operations.
    pub struct Primary;
    impl QueueType for Primary {}
    // Transfer-only DMA queue (optionally supported).
    pub struct AsyncTransfer;
    impl QueueType for AsyncTransfer {}
    // Async compute-only queue (optionally supported).
    pub struct AsyncCompute;
    impl QueueType for AsyncCompute {}
}
use queue_type::*;

pub struct Queue<T: QueueType> {
    handle: vk::Queue,
    family_index: u32,
    phantom_data: PhantomData<T>,
}

impl<T: QueueType> Queue<T> {
    pub fn get(device: &ash::Device, queue_family_idx: u32, queue_idx: u32) -> Self {
        let handle = unsafe { device.get_device_queue(queue_family_idx, queue_idx) };
        Self {
            handle,
            family_index: queue_family_idx,
            phantom_data: PhantomData,
        }
    }

    pub fn handle(&self) -> vk::Queue {
        self.handle
    }

    pub fn family_index(&self) -> u32 {
        self.family_index
    }
}
