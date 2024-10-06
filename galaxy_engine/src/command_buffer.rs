use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::device::{Device, SharedDeviceLoader};

// Either use a specific command buffer, or allocate a new one-time buffer from a pool.
// TODO: unify queue / pool system.
pub enum TransientOrPersistentCommandBuffer<'a> {
    Persistent(&'a CommandBuffer),
    Transient(CommandBuffer),
}

impl<'a> TransientOrPersistentCommandBuffer<'a> {
    pub fn command_buffer(&self) -> &CommandBuffer {
        match self {
            Self::Persistent(cmd_buf) => cmd_buf,
            Self::Transient(cmd_buf) => cmd_buf,
        }
    }
    pub fn maybe_end_submit_and_wait(self, device: &Device, queue: vk::Queue) -> VkResult<()> {
        match self {
            Self::Persistent(_) => Ok(()),
            Self::Transient(cmd_buf) => cmd_buf.end_submit_and_wait(device, queue),
        }
    }
}

pub struct CommandBuffer {
    loader: SharedDeviceLoader,
    handle: vk::CommandBuffer,
    pool: vk::CommandPool,
}

impl CommandBuffer {
    pub fn one_time_transient(device: &Device, cmd_pool: vk::CommandPool) -> VkResult<TransientOrPersistentCommandBuffer> {
        Ok(TransientOrPersistentCommandBuffer::Transient(Self::begin_one_time(device, cmd_pool)?))
    }
    pub fn begin_one_time(device: &Device, cmd_pool: vk::CommandPool) -> VkResult<Self> {
        let cmd_buffer = unsafe { device.allocate_command_buffer(cmd_pool, vk::CommandBufferLevel::PRIMARY) }?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.loader().begin_command_buffer(cmd_buffer, &begin_info) }?;
        Ok(Self {
            loader: device.cloned_loader(),
            handle: cmd_buffer,
            pool: cmd_pool,
        })
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn end_and_submit(&self, device: &Device, queue: vk::Queue) -> VkResult<()> {
        unsafe { device.loader().end_command_buffer(self.handle) }?;
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&self.handle));
        unsafe { device.loader().queue_submit(queue, &[submit_info], vk::Fence::null()) }?;
        Ok(())
    }

    pub fn end_submit_and_wait(self, device: &Device, queue: vk::Queue) -> VkResult<()> {
        self.end_and_submit(device, queue)?;
        unsafe { device.loader().queue_wait_idle(queue) }?;
        // Command buffer is freed in Drop.
        Ok(())
    }

    pub fn as_persistent(&self) -> TransientOrPersistentCommandBuffer {
        TransientOrPersistentCommandBuffer::Persistent(self)
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe { self.loader.free_command_buffers(self.pool, &[self.handle]) };
    }
}

