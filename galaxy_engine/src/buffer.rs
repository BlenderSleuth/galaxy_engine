use std::slice;
use ash::prelude::VkResult;
use ash::vk;

use crate::device::{Device, QueueFamily};

pub fn copy_buffer(cmd_pool: vk::CommandPool, device: &Device, src_buffer: &Buffer, dst_buffer: &Buffer, size: vk::DeviceSize) -> VkResult<()> {
    let cmd_buffer = unsafe { device.allocate_command_buffer(cmd_pool, vk::CommandBufferLevel::PRIMARY) }?;
    
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.device().begin_command_buffer(cmd_buffer, &begin_info) }?;
    
    let copy_region = vk::BufferCopy::default()
        .size(size);
    unsafe { device.device().cmd_copy_buffer(cmd_buffer, src_buffer.handle(), dst_buffer.handle(), &[copy_region]) };
    
    unsafe { device.device().end_command_buffer(cmd_buffer) }?;
    
    let transfer_queue = device.get_queue(QueueFamily::Transfer);
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(slice::from_ref(&cmd_buffer));
    unsafe { device.device().queue_submit(transfer_queue, &[submit_info], vk::Fence::null()) }?;
    unsafe { device.device().queue_wait_idle(transfer_queue) }?;
    
    unsafe { device.device().free_command_buffers(cmd_pool, slice::from_ref(&cmd_buffer)) };
    
    Ok(())
}

pub struct Buffer {
    handle: vk::Buffer,
    mem_requirements: vk::MemoryRequirements,
    memory: vk::DeviceMemory,
    length: usize,
}

impl Buffer {
    pub fn new_for_typed_data<T: bytemuck::Pod>(device: &Device, data: &[T], usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        Self::new_for_data(device, bytemuck::cast_slice(data), data.len(), usage, sharing_mode, memory_properties)
    }

    pub fn new_for_data(device: &Device, data: &[u8], length: usize, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        Self::new(device, std::mem::size_of_val(data) as vk::DeviceSize, length, usage, sharing_mode, memory_properties)
    }
    
    pub fn new(device: &Device, size: vk::DeviceSize, length: usize, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        let vertex_buffer_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(sharing_mode);
        let handle = unsafe { device.device().create_buffer(&vertex_buffer_info, None) }?;

        // Allocate memory for buffer.
        let mem_requirements = unsafe { device.device().get_buffer_memory_requirements(handle) };
        let memory_type = device.find_memory_type(mem_requirements.memory_type_bits, memory_properties);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type.unwrap());
        let memory = unsafe { device.device().allocate_memory(&alloc_info, None) }?;

        // Bind buffer memory.
        unsafe { device.device().bind_buffer_memory(handle, memory, 0) }?;

        
        Ok(Self { handle, mem_requirements, memory, length })
    }
    
    pub fn copy_into_buffer<T: bytemuck::Pod>(&mut self, device: &Device, typed_data: &[T]) -> VkResult<()> {
        let data_size = std::mem::size_of_val(typed_data);
        let data = bytemuck::cast_slice(typed_data);
        {
            // Map memory.
            let mem_ptr = unsafe { device.device().map_memory(self.memory, 0, data_size as vk::DeviceSize, vk::MemoryMapFlags::empty()) }?;
            // Copy data to buffer.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), mem_ptr as *mut u8, data_size) };
            // Unmap memory.
            unsafe { device.device().unmap_memory(self.memory) };
        }
        
        Ok(())
    }
    
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }
    
    // The size of the buffer in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.mem_requirements.size
    }
    
    // The number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.length
    }
    
    pub unsafe fn destroy(&self, device: &Device) {
        device.device().destroy_buffer(self.handle, None);
        device.device().free_memory(self.memory, None);
    }
}
