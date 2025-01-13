// Copyright (c) 2025 Ben Sutherland.

use ash::prelude::VkResult;

use crate::vulkan::buffer::StagingBuffer;
use crate::vulkan::command_buffer::{
    CommandPool, ExecutableCmdBuf, PendingCmdBuf, RecordingCmdBuf, SubmitInfo, Transient,
};
use crate::vulkan::device::queue::queue_type::QueueType;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::queue::queue_type::PrimaryQueue;

pub struct LoadingContext<'a, Q: QueueType = PrimaryQueue> {
    device: &'a Device,
    cmd_pool: &'a mut CommandPool<Q, Transient>,
    executable: Vec<ExecutableCmdBuf<Q>>,
    pending: Vec<PendingCmdBuf<Q>>,
    staging_buffers: Vec<StagingBuffer>,
}

impl<'a, Q: QueueType> LoadingContext<'a, Q> {
    pub fn new(device: &'a Device, cmd_pool: &'a mut CommandPool<Q, Transient>) -> VkResult<Self> {
        Ok(Self {
            device,
            cmd_pool,
            executable: Vec::new(),
            pending: Vec::new(),
            staging_buffers: Vec::new(),
        })
    }

    pub fn load<R: IntoIterator<Item = StagingBuffer>>(
        &mut self,
        loading_func: impl FnOnce(&mut RecordingCmdBuf<Q>) -> MemResult<R>,
    ) -> MemResult<()> {
        let mut recording = self.cmd_pool.allocate_transient_cmd_buffer()?;
        self.staging_buffers.extend(loading_func(&mut recording)?.into_iter());
        self.executable.push(recording.end()?);
        Ok(())
    }

    pub fn submit(&mut self) -> VkResult<()> {
        if self.executable.is_empty() {
            return Ok(());
        }

        let submit_info = SubmitInfo {
            cmd_buffers: &mut self.executable,
            wait_semaphores: &[],
            signal_semaphores: &[],
        };
        ExecutableCmdBuf::submit(self.device, [submit_info], None, &mut self.pending)?;
        assert!(self.executable.is_empty());
        Ok(())
    }

    pub fn complete(mut self) -> VkResult<()> {
        self.submit()?;
        PendingCmdBuf::wait_idle(std::mem::take(&mut self.pending), self.device)?;
        Ok(())
    }
}

impl<'a, Q: QueueType> Drop for LoadingContext<'a, Q> {
    fn drop(&mut self) {
        if !self.executable.is_empty() {
            log::error!("Dropping LoadingContext with unsubmitted commands.");
        }
        if !self.pending.is_empty() {
            log::error!("Dropping LoadingContext with uncompleted commands. Leaking required resources.");
            std::mem::take(&mut self.staging_buffers).leak();
            // Dropping pending commands leaks them.
            self.pending.clear();
        }
    }
}
