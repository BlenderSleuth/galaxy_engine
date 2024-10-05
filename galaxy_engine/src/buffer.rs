use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::buffer::mem_location::*;
use crate::command_buffer::CommandBuffer;
use crate::device::{Device, QueueFamily};
use crate::engine::{MemResult, MemoryError};
use crate::utils;

pub mod mem_location {
    use gpu_allocator::MemoryLocation;

    // Type-state trait encoding of gpu_allocator::MemoryLocation for use in generic parameters.
    pub trait MemLocation {
        fn location() -> MemoryLocation;
    }
    pub enum Unknown {}
    impl MemLocation for Unknown {
        fn location() -> MemoryLocation { MemoryLocation::Unknown }
    }
    pub enum GpuOnly {}
    impl MemLocation for GpuOnly {
        fn location() -> MemoryLocation { MemoryLocation::GpuOnly }
    }
    pub enum CpuToGpu {}
    impl MemLocation for CpuToGpu {
        fn location() -> MemoryLocation { MemoryLocation::CpuToGpu }
    }
    pub enum GpuToCpu {}
    impl MemLocation for GpuToCpu {
        fn location() -> MemoryLocation { MemoryLocation::GpuToCpu }
    }
}

pub struct Buffer<L: MemLocation> {
    handle: vk::Buffer,
    allocation: Option<Allocation>,
    length: u32,
    element_size: vk::DeviceSize,
    _mem_location: std::marker::PhantomData<L>,
}

impl<L: MemLocation> Buffer<L> {
    // NOTE: When buffers are being used for multiple resources, should we remove the length and element size fields?
    pub fn new_for_typed_data<T: bytemuck::Pod>(device: &Device, name: &str, data: &[T], usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode) -> MemResult<Self> {
        Self::new(device, name, data.len() as u32, std::mem::size_of::<T>(), usage, sharing_mode)
    }

    pub fn new(device: &Device, name: &str, length: u32, element_size: usize, usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode) -> MemResult<Self> {
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

        // Allocate memory for buffer. Check if the buffer requires dedicated allocation.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default()
            .push_next(&mut dedicated_requirements);
        let requirements_info = vk::BufferMemoryRequirementsInfo2::default()
            .buffer(handle);
        unsafe { device.device().get_buffer_memory_requirements2(&requirements_info, &mut requirements) };

        let requirements = requirements.memory_requirements;
        let allocation_scheme = if utils::use_dedicated_allocation(dedicated_requirements) {
            AllocationScheme::DedicatedBuffer(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let desc = AllocationCreateDesc {
            name,
            requirements,
            location: L::location(),
            linear: true,
            allocation_scheme,
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

    pub unsafe fn destroy(&mut self, device: &Device) -> MemResult<()> {
        device.device().destroy_buffer(self.handle, None);
        if let Some(allocation) = self.allocation.take() {
            device.free_memory(allocation)
        } else {
            Ok(())
        }
    }

    pub fn copy_to_buffer<L2: MemLocation>(&self, cmd_pool: vk::CommandPool, device: &Device, dst_buffer: &mut Buffer<L2>, size: vk::DeviceSize, queue_family: QueueFamily) -> MemResult<()> {
        let cmd_buffer = CommandBuffer::begin_one_time(device, cmd_pool)?;

        let copy_region = vk::BufferCopy::default()
            .size(size);
        unsafe { device.device().cmd_copy_buffer(cmd_buffer.handle(), self.handle(), dst_buffer.handle(), &[copy_region]) };

        Ok(cmd_buffer.end_and_submit(device.get_queue(queue_family))?)
    }
}

impl Buffer<GpuOnly> {
    pub fn copy_via_staging_buffer(&mut self, device: &Device, transfer_cmd_pool: vk::CommandPool, src_data: &[u8]) -> MemResult<()> {
        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            &device,
            "Staging Buffer",
            src_data.len() as u32,
            std::mem::size_of::<u8>(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        staging_buffer.copy_into_buffer(&src_data)?;
        staging_buffer.copy_to_buffer(transfer_cmd_pool, &device, self, staging_buffer.size(), QueueFamily::Transfer)?;
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