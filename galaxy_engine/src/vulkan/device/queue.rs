// Copyright (c) 2024-2025 Ben Sutherland.

use std::marker::PhantomData;
use std::sync::Mutex;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;
use itertools::izip;

pub mod queue_type {
    // We need a graphics queue for rendering, and we also get a "sync" compute queue for free.
    // From the spec: "If an implementation exposes any queue family that supports graphics operations,
    // at least one queue family of at least one physical vulkan exposed by the implementation must
    // support both graphics and compute operations."

    // Additionally from the spec: "All commands that are allowed on a queue that supports transfer
    // operations are also allowed on a queue that supports either graphics or compute operations."
    // So we can use the graphics queue for "sync" transfer operations as well.

    // If we end up using the separate queue types, it might be worth creating more primary queues
    // as fill-ins if the specialised ones are not supported. This means we can keep the async
    // architecture of the async compute and transfer threads.

    // Not from the spec, but on good authority, "No such hardware exists" that supports graphics and
    // presentation without a graphics and present queue. Thus, this primary queue is also used for presentation.
    // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th

    // For MoltenVK, we may want to enable the MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES option so the
    // specialised queues get created (would need to be profiled).

    use crate::vulkan::device::GetQueue;

    pub trait QueueType: Sized + GetQueue + 'static {}
    pub trait ComputeQueueType: QueueType {}
    // This is the primary, mandatory queue, with support for graphics, compute, transfer and present operations.
    pub struct PrimaryQueue;
    impl QueueType for PrimaryQueue {}
    impl ComputeQueueType for PrimaryQueue {}

    // Transfer-only DMA queue (optionally supported).
    pub struct AsyncTransferQueue;
    impl QueueType for AsyncTransferQueue {}

    // Async compute-only queue (optionally supported).
    pub struct AsyncComputeQueue;
    impl QueueType for AsyncComputeQueue {}
    impl ComputeQueueType for AsyncComputeQueue {}
}
pub use queue_type::QueueType;

use crate::vulkan::command_buffer::ExecutableCmdBuf;
use crate::vulkan::device::physical_device::QueueArray;
use crate::vulkan::device::SharedDeviceLoader;
use crate::vulkan::queue::queue_type::PrimaryQueue;
use crate::vulkan::swapchain::{Swapchain, SwapchainImage};
use crate::vulkan::sync::Fence;

// A non-thread-safe handle to a queue.
//#[derive(Copy, Clone)]
//struct QueueHandle(vk::Queue, PhantomData<*const ()>);

pub struct WaitSemaphore {
    pub handle: vk::Semaphore,
    pub stage_mask: vk::PipelineStageFlags,
}

pub struct SubmitInfo<'a, Q: QueueType> {
    pub cmd_buffers: &'a [ExecutableCmdBuf<Q>],
    pub wait_semaphores: &'a [WaitSemaphore],
    pub signal_semaphores: &'a [vk::Semaphore],
}

pub struct Queue<Q: QueueType> {
    loader: SharedDeviceLoader,
    handle: vk::Queue,
    pub family_index: u32,
    pub index: u32,
    _ty: PhantomData<Q>,
}

impl<Q: QueueType> Queue<Q> {
    // Safety: The queue type must match the type of the queue family.
    // Will panic if a queue of the same family and index has already been created.
    pub unsafe fn get(loader: SharedDeviceLoader, queue_family_idx: u32, queue_index: u32) -> Self {
        {
            // Only one queue of a given family and index should be created, to ensure each queue is only accessed by one thread at a time.
            static CREATED_QUEUES: Mutex<QueueArray<(u32, u32)>> = Mutex::new(ArrayVec::new_const());

            let mut created_queues = CREATED_QUEUES.lock().unwrap();
            let queue_key = (queue_family_idx, queue_index);
            if created_queues.contains(&queue_key) {
                panic!("Queue (family: {queue_family_idx}, index: {queue_index}) already created");
            } else {
                created_queues.push(queue_key);
            }
        }

        let handle = unsafe { loader.get_device_queue(queue_family_idx, queue_index) };
        Self {
            loader,
            handle,
            family_index: queue_family_idx,
            index: queue_index,
            _ty: PhantomData,
        }
    }

    pub(super) fn vk_handle(&self) -> vk::Queue {
        self.handle
    }

    pub(crate) unsafe fn submit(&self, submit_infos: &[SubmitInfo<Q>], fence: Option<&Fence>) -> VkResult<()> {
        fn calculate_offsets<Q: QueueType>(
            submit_infos: &[SubmitInfo<Q>],
            mut f: impl FnMut(&SubmitInfo<Q>) -> usize,
        ) -> Vec<usize> {
            submit_infos
                .iter()
                .scan(0, |len, submit_info| {
                    let offset = *len;
                    *len += f(submit_info);
                    Some(offset)
                })
                .collect()
        }

        let wait_semaphore_offsets = calculate_offsets(submit_infos, |submit_info| submit_info.wait_semaphores.len());

        // Unzip wait semaphores.
        let (wait_semaphore_handles, wait_semaphore_stage_masks): (Vec<_>, Vec<_>) = submit_infos
            .iter()
            .flat_map(|submit_info| submit_info.wait_semaphores.iter().map(|s| (s.handle, s.stage_mask)))
            .unzip();

        // Get cmd buffer handles.
        let cmd_buffer_offsets = calculate_offsets(submit_infos, |submit_info| submit_info.cmd_buffers.len());
        let cmd_buffers: Vec<_> = submit_infos
            .iter()
            .flat_map(|submit_info| submit_info.cmd_buffers.iter().map(|c| c.handle()))
            .collect();

        let vk_submit_infos: Vec<_> = izip!(submit_infos.iter(), wait_semaphore_offsets, cmd_buffer_offsets)
            .map(|(submit_info, wait_offset, cmd_buffer_offset)| {
                let wait_range = wait_offset..wait_offset + submit_info.wait_semaphores.len();
                let cmd_buffer_range = cmd_buffer_offset..cmd_buffer_offset + submit_info.cmd_buffers.len();
                vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphore_handles[wait_range.clone()])
                    .wait_dst_stage_mask(&wait_semaphore_stage_masks[wait_range])
                    .signal_semaphores(submit_info.signal_semaphores)
                    .command_buffers(&cmd_buffers[cmd_buffer_range])
            })
            .collect();

        unsafe {
            self.loader.queue_submit(
                self.handle,
                &vk_submit_infos,
                fence.map(|f| f.handle()).unwrap_or(vk::Fence::null()),
            )
        }
    }

    pub fn wait_idle(&self) -> VkResult<()> {
        unsafe { self.loader.queue_wait_idle(self.handle) }
    }
}

impl Queue<PrimaryQueue> {
    pub fn present(
        &self,
        swapchain: &Swapchain,
        swapchain_image: SwapchainImage,
        wait_semaphores: &[vk::Semaphore],
    ) -> VkResult<bool> {
        unsafe { swapchain.queue_present(self.handle, swapchain_image, wait_semaphores) }
    }
}

// GetQueue allows us to get a queue from the device with just the typestate generic.
//b trait GetQueue: Sized + QueueType {
//  fn get_queue(device: &mut Device) -> Option<&mut Queue<Self>>;
//
//
//pl GetQueue for PrimaryQueue {
//  fn get_queue(device: &mut Device) -> Option<&mut Queue<Self>> {
//      Some(device.primary_queue())
//  }
//
//
//pl GetQueue for AsyncTransferQueue {
//  fn get_queue(device: &mut Device) -> Option<&mut Queue<Self>> {
//      device.async_transfer_queue()
//  }
//
//
//pl GetQueue for AsyncComputeQueue {
//  fn get_queue(device: &mut Device) -> Option<&mut Queue<Self>> {
//      device.async_compute_queue()
//  }
//
