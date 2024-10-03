use std::slice;
use ash::prelude::VkResult;
use ash::vk;
use crate::device::Device;

pub struct CommandBuffer<'a> {
    handle: vk::CommandBuffer,
    device: &'a Device,
    pool: vk::CommandPool,
}

impl <'a> CommandBuffer<'a> {
    pub fn begin_one_time(device: &'a Device, cmd_pool: vk::CommandPool) -> VkResult<Self> {
        let cmd_buffer = unsafe { device.allocate_command_buffer(cmd_pool, vk::CommandBufferLevel::PRIMARY) }?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.device().begin_command_buffer(cmd_buffer, &begin_info) }?;
        Ok(Self {
            handle: cmd_buffer,
            device,
            pool: cmd_pool,
        })
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn end_and_submit(self, queue: vk::Queue) -> VkResult<()> {
        unsafe { self.device.device().end_command_buffer(self.handle) }?;
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&self.handle));
        unsafe { self.device.device().queue_submit(queue, &[submit_info], vk::Fence::null()) }?;
        unsafe { self.device.device().queue_wait_idle(queue) }?;
        // Command buffer is freed in Drop.
        Ok(())
    }
}

impl Drop for CommandBuffer<'_> {
    fn drop(&mut self) {
        unsafe { self.device.device().free_command_buffers(self.pool, &[self.handle]) };
    }
}

