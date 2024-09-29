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

pub fn copy_via_staging_buffer<T: bytemuck::Pod>(device: &Device, transfer_cmd_pool: vk::CommandPool, src_data: &[T], dst_buffer: &Buffer) -> VkResult<()> {
    let mut staging_buffer = Buffer::new_for_typed_data(
        &device,
        &src_data,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::SharingMode::EXCLUSIVE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging_buffer.copy_into_buffer(&device, &src_data)?;
    copy_buffer(transfer_cmd_pool, &device, &staging_buffer, &dst_buffer, staging_buffer.size())?;
    unsafe { staging_buffer.destroy(&device) };
    Ok(())
}

pub struct Buffer {
    handle: vk::Buffer,
    mem_requirements: vk::MemoryRequirements,
    memory: vk::DeviceMemory,
    length: u32,
}

impl Buffer {
    pub fn new_for_typed_data<T: bytemuck::Pod>(device: &Device, data: &[T], usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        Self::new_for_data(device, bytemuck::cast_slice(data), data.len() as u32, usage, sharing_mode, memory_properties)
    }

    pub fn new_for_data(device: &Device, data: &[u8], length: u32, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        Self::new(device, std::mem::size_of_val(data) as vk::DeviceSize, length, usage, sharing_mode, memory_properties)
    }

    pub fn new(device: &Device, size: vk::DeviceSize, length: u32, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode, memory_properties: vk::MemoryPropertyFlags) -> Result<Self, vk::Result> {
        let device_properties = device.get_properties();
        let queue_indices = [device.get_properties().graphics_queue_family_idx, device_properties.transfer_queue_family_idx];

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(if sharing_mode == vk::SharingMode::CONCURRENT { &queue_indices } else { &[] });
        let handle = unsafe { device.device().create_buffer(&buffer_info, None) }?;

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
        assert_eq!(data_size, self.size() as usize);
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
    pub fn len(&self) -> u32 {
        self.length
    }
    
    // TODO: offset and size should be optional together.
    pub fn map(&self, device: &Device, offset: vk::DeviceSize, size: Option<vk::DeviceSize>) -> VkResult<*mut u8> {
        unsafe { device.device().map_memory(self.memory, offset, size.unwrap_or(vk::WHOLE_SIZE), vk::MemoryMapFlags::empty()) }.map(|ptr| ptr as *mut u8)
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.device().destroy_buffer(self.handle, None);
        device.device().free_memory(self.memory, None);
    }
}
