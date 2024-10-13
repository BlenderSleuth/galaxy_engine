use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};

use crate::command_buffer::{CommandBuffer, TransientOrPersistentCommandBuffer};
use crate::device::{Device, QueueFamily, SharedDeviceLoader};
use crate::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::{debug, gpu_alloc};

use gpu_allocator::MemoryLocation;

// Type-state trait encoding of gpu_allocator::MemoryLocation for use in generic parameters.
pub trait MemLocation {
    fn location() -> MemoryLocation;
}
pub struct Unknown;
impl MemLocation for Unknown {
    fn location() -> MemoryLocation { MemoryLocation::Unknown }
}
pub struct GpuOnly;
impl MemLocation for GpuOnly {
    fn location() -> MemoryLocation { MemoryLocation::GpuOnly }
}
pub struct CpuToGpu;
impl MemLocation for CpuToGpu {
    fn location() -> MemoryLocation { MemoryLocation::CpuToGpu }
}
pub struct GpuToCpu;
impl MemLocation for GpuToCpu {
    fn location() -> MemoryLocation { MemoryLocation::GpuToCpu }
}

pub struct Buffer<L: MemLocation> {
    loader: SharedDeviceLoader,
    alloc: SharedAllocator,
    handle: vk::Buffer,
    allocation: ManuallyFreeAllocation,
    // TODO: Remove length and element size fields when buffers are used for multiple resources.
    length: u32,
    element_size: vk::DeviceSize,
    _mem_location: std::marker::PhantomData<L>,
}

impl<L: MemLocation> Buffer<L> {
    pub fn new_for_typed_data<T: bytemuck::Pod>(name: &str, device: &Device, data: &[T], usage: vk::BufferUsageFlags, sharing_mode: vk::SharingMode) -> MemResult<Self> {
        Self::new(name, device, data.len() as u32, std::mem::size_of::<T>(), usage, sharing_mode)
    }

    pub fn new(
        name: &str,
        device: &Device,
        length: u32,
        element_size: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
    ) -> MemResult<Self> {
        let device_properties = device.get_properties();
        let queue_indices = [device.get_properties().graphics_queue_family_idx, device_properties.transfer_queue_family_idx];

        let element_size = element_size as vk::DeviceSize;
        let size = element_size * length as vk::DeviceSize;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(if sharing_mode == vk::SharingMode::CONCURRENT { &queue_indices } else { &[] });
        let handle = unsafe { device.loader().create_buffer(&buffer_info, None) }?;

        // Debug name object.
        debug::set_object_name(device, handle, name)?;

        // Allocate memory for buffer. Check if the buffer requires dedicated allocation.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default()
            .push_next(&mut dedicated_requirements);
        let requirements_info = vk::BufferMemoryRequirementsInfo2::default()
            .buffer(handle);
        unsafe { device.loader().get_buffer_memory_requirements2(&requirements_info, &mut requirements) };

        let requirements = requirements.memory_requirements;
        let allocation_scheme = if gpu_alloc::use_dedicated_allocation(dedicated_requirements) {
            AllocationScheme::DedicatedBuffer(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let desc = AllocationCreateDesc {
            name: debug::debug_only_name!(name),
            requirements,
            location: L::location(),
            linear: true,
            allocation_scheme,
        };
        let allocation = device.allocate_and_bind_memory(&desc, handle)?;

        Ok(Self { loader: device.cloned_loader(), alloc: device.cloned_allocator(), handle, allocation, length, element_size, _mem_location: std::marker::PhantomData })
    }

    pub fn handle(&self) -> vk::Buffer { self.handle }

    // The size of the buffer in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.element_size * self.length as vk::DeviceSize
    }

    // The number of elements that can be stored in the buffer.
    pub fn len(&self) -> u32 { self.length }

    // Descriptor buffer info for the whole buffer.
    pub fn descriptor_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.handle())
            .offset(0)
            .range(self.size())
    }

    pub fn copy_to_buffer<L2: MemLocation>(&self, cmd: TransientOrPersistentCommandBuffer, device: &Device, dst_buffer: &mut Buffer<L2>, size: vk::DeviceSize, queue_family: QueueFamily) -> MemResult<()> {
        let cmd_buffer = cmd.command_buffer();

        let copy_region = vk::BufferCopy::default()
            .size(size);
        unsafe { device.loader().cmd_copy_buffer(cmd_buffer.handle(), self.handle(), dst_buffer.handle(), &[copy_region]) };

        Ok(cmd.maybe_end_submit_and_wait(device, device.get_queue(queue_family))?)
    }
}

impl<L: MemLocation> Drop for Buffer<L> {
    fn drop(&mut self) {
        // Drop buffer.
        unsafe { self.loader.destroy_buffer(self.handle, None); }
        unsafe { gpu_alloc::free_or_log_on_fail(&self.alloc, &mut self.allocation) };
    }
}

impl Buffer<GpuOnly> {
    pub fn copy_via_staging_buffer(&mut self, device: &Device, src_data: &[u8], cmd_pool: vk::CommandPool, queue_family: QueueFamily) -> MemResult<()> {
        let mut staging_buffer = Buffer::<CpuToGpu>::new(
            "Staging Buffer",
            &device,
            src_data.len() as u32,
            std::mem::size_of::<u8>(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        staging_buffer.copy_into_buffer(&src_data, 0)?;
        staging_buffer.copy_to_buffer(CommandBuffer::one_time_transient(device, cmd_pool)?, &device, self, staging_buffer.size(), queue_family)?;
        Ok(())
    }
}

impl Buffer<CpuToGpu> {
    pub fn copy_into_buffer(&mut self, data: &[u8], offset: usize) -> MemResult<()> {
        // CPU to GPU memory is always mappable.
        let mut memory = self.allocation.try_as_mapped_slab().unwrap();
        presser::copy_from_slice_to_offset_with_align(data, &mut memory, offset, 1)?;
        Ok(())
    }
}