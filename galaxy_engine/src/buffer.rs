use std::slice;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::buffer::mem_location::*;
use crate::device::{Device, QueueFamily};
use crate::engine::{MemResult, MemoryError};

pub mod mem_location {
    use gpu_allocator::MemoryLocation;

    // Type-state trait encoding of gpu_allocator::MemoryLocation for use in generic parameters.
    pub trait MemLocationTrait {
        fn location() -> MemoryLocation;
    }
    pub enum Unknown {}
    impl MemLocationTrait for Unknown {
        fn location() -> MemoryLocation { MemoryLocation::Unknown }
    }
    pub enum GpuOnly {}
    impl MemLocationTrait for GpuOnly {
        fn location() -> MemoryLocation { MemoryLocation::GpuOnly }
    }
    pub enum CpuToGpu {}
    impl MemLocationTrait for CpuToGpu {
        fn location() -> MemoryLocation { MemoryLocation::CpuToGpu }
    }
    pub enum GpuToCpu {}
    impl MemLocationTrait for GpuToCpu {
        fn location() -> MemoryLocation { MemoryLocation::GpuToCpu }
    }
}

pub struct Buffer<L: MemLocationTrait> {
    handle: vk::Buffer,
    allocation: Option<Allocation>,
    length: u32,
    element_size: vk::DeviceSize,
    _mem_location: std::marker::PhantomData<L>,
}

impl<L: MemLocationTrait> Buffer<L> {
    pub fn new_for_typed_data<T: bytemuck::Pod>(device: &Device, data: &[T], usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode) -> MemResult<Self> {
        Self::new(device, data.len() as u32, std::mem::size_of::<T>(), usage, sharing_mode)
    }

    pub fn new(device: &Device, length: u32, element_size: usize, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode) -> MemResult<Self> {
        let device_properties = device.get_properties();
        let queue_indices = [device.get_properties().graphics_queue_family_idx, device_properties.transfer_queue_family_idx];

        let element_size = element_size as vk::DeviceSize;
        let size = element_size * length as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(if sharing_mode == vk::SharingMode::CONCURRENT { &queue_indices } else { &[] });
        let handle = unsafe { device.device().create_buffer(&buffer_info, None) }?;

        // Allocate memory for buffer.
        let requirements = unsafe { device.device().get_buffer_memory_requirements(handle) };
        let desc = AllocationCreateDesc {
            name: "Buffer Allocation",
            requirements,
            location: L::location(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = device.allocate_memory(&desc)?;

        // Bind buffer memory.
        unsafe { device.device().bind_buffer_memory(handle, allocation.memory(), allocation.offset()) }?;

        Ok(Self { handle, allocation: Some(allocation), length, element_size, _mem_location: std::marker::PhantomData })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    // The size of the buffer in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.element_size * self.length as vk::DeviceSize
    }

    // The number of elements that can be stored in the buffer.
    pub fn len(&self) -> u32 {
        self.length
    }

    //pub fn map(&self, device: &Device, offset: vk::DeviceSize, size: Option<vk::DeviceSize>) -> VkResult<*mut u8> {
    //    unsafe { device.device().map_memory(self.memory, offset, size.unwrap_or(vk::WHOLE_SIZE), vk::MemoryMapFlags::empty()) }.map(|ptr| ptr as *mut u8)
    //}

    // If size is none, will map entire allocation.
    //pub fn map_guard(&self, device: &Device, offset: vk::DeviceSize, size: Option<vk::DeviceSize>) -> VkResult<MappedMemoryGuard> {
    //    MappedMemoryGuard::new(device, self.memory, offset, size.unwrap_or(self.mem_requirements.size))
    //}

    //pub fn unmap(&self, device: &Device) {
    //    unsafe { device.device().unmap_memory(self.memory) };
    //}

    pub unsafe fn destroy(&mut self, device: &Device) -> MemResult<()> {
        device.device().destroy_buffer(self.handle, None);
        if let Some(allocation) = self.allocation.take() {
            device.free_memory(allocation)
        } else {
            Ok(())
        }
    }

    pub fn copy_to_buffer<L2: MemLocationTrait>(&self, cmd_pool: vk::CommandPool, device: &Device, dst_buffer: &mut Buffer<L2>, size: vk::DeviceSize) -> MemResult<()> {
        let cmd_buffer = unsafe { device.allocate_command_buffer(cmd_pool, vk::CommandBufferLevel::PRIMARY) }?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.device().begin_command_buffer(cmd_buffer, &begin_info) }?;

        let copy_region = vk::BufferCopy::default()
            .size(size);
        unsafe { device.device().cmd_copy_buffer(cmd_buffer, self.handle(), dst_buffer.handle(), &[copy_region]) };

        unsafe { device.device().end_command_buffer(cmd_buffer) }?;

        let transfer_queue = device.get_queue(QueueFamily::Transfer);
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&cmd_buffer));
        unsafe { device.device().queue_submit(transfer_queue, &[submit_info], vk::Fence::null()) }?;
        unsafe { device.device().queue_wait_idle(transfer_queue) }?;

        unsafe { device.device().free_command_buffers(cmd_pool, slice::from_ref(&cmd_buffer)) };

        Ok(())
    }
}

impl Buffer<GpuOnly> {
    pub fn copy_via_staging_buffer(&mut self, device: &Device, transfer_cmd_pool: vk::CommandPool, src_data: &[u8]) -> MemResult<()> {
        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            &device,
            src_data.len() as u32,
            std::mem::size_of::<u8>(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        staging_buffer.copy_into_buffer(&src_data)?;
        staging_buffer.copy_to_buffer(transfer_cmd_pool, &device, self, staging_buffer.size())?;
        unsafe { staging_buffer.destroy(&device) }?;
        Ok(())
    }
}

impl Buffer<CpuToGpu> {
    pub fn copy_into_buffer(&mut self, data: &[u8]) -> MemResult<()> {
        let allocation = self.allocation.as_mut().ok_or(MemoryError::NotAllocated("Buffer"))?;
        // CPU to GPU memory is always mappable.
        let mut memory = allocation.try_as_mapped_slab().unwrap();
        presser::copy_from_slice_to_offset_with_align(data, &mut memory, 0, 1)?;
        Ok(())
    }
}