// Copyright (c) 2024 Ben Sutherland.

use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::vulkan::command_buffer::RecordingCmdBuf;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::vulkan::queue::queue_type::QueueType;
use crate::vulkan::{debug, gpu_alloc};

// Type-state trait encoding of gpu_allocator::MemoryLocation for use in generic parameters.
pub trait MemLocation {
    fn location() -> MemoryLocation;
}
pub struct Unknown;
impl MemLocation for Unknown {
    fn location() -> MemoryLocation {
        MemoryLocation::Unknown
    }
}
pub struct GpuOnly;
impl MemLocation for GpuOnly {
    fn location() -> MemoryLocation {
        MemoryLocation::GpuOnly
    }
}
pub struct CpuToGpu;
impl MemLocation for CpuToGpu {
    fn location() -> MemoryLocation {
        MemoryLocation::CpuToGpu
    }
}
pub struct GpuToCpu;
impl MemLocation for GpuToCpu {
    fn location() -> MemoryLocation {
        MemoryLocation::GpuToCpu
    }
}

pub struct Buffer<L: MemLocation> {
    loader: SharedDeviceLoader,
    alloc: SharedAllocator,
    handle: vk::Buffer,
    allocation: ManuallyFreeAllocation,
    size: vk::DeviceSize,
    _mem_location: std::marker::PhantomData<L>,
}

impl<L: MemLocation> Buffer<L> {
    pub fn new_for_type<T: bytemuck::Pod>(name: &str, device: &Device, usage: vk::BufferUsageFlags) -> MemResult<Self> {
        Self::new(name, device, std::mem::size_of::<T>() as vk::DeviceSize, usage)
    }

    pub fn new_for_slice<T: bytemuck::Pod>(
        name: &str,
        device: &Device,
        slice: &[T],
        usage: vk::BufferUsageFlags,
    ) -> MemResult<Self> {
        Self::new(name, device, std::mem::size_of_val(slice) as vk::DeviceSize, usage)
    }

    pub fn new(name: &str, device: &Device, size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> MemResult<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.loader().create_buffer(&buffer_info, None) }?;

        // Debug name object.
        debug::set_object_name(device, handle, name)?;

        // Allocate memory for buffer. Check if the buffer requires dedicated allocation.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        let requirements_info = vk::BufferMemoryRequirementsInfo2::default().buffer(handle);
        unsafe {
            device
                .loader()
                .get_buffer_memory_requirements2(&requirements_info, &mut requirements)
        };

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

        Ok(Self {
            loader: device.cloned_loader(),
            alloc: device.cloned_allocator(),
            handle,
            allocation,
            size,
            _mem_location: std::marker::PhantomData,
        })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    // The size of the buffer in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    // Descriptor buffer info for the whole buffer.
    pub fn descriptor_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.handle())
            .offset(0)
            .range(self.size())
    }

    pub fn copy_to_buffer<L2: MemLocation>(
        &self,
        cmd_buffer: &mut RecordingCmdBuf<impl QueueType>,
        dst_buffer: &mut Buffer<L2>,
        size: vk::DeviceSize,
    ) {
        let copy_region = vk::BufferCopy::default().size(size);
        cmd_buffer.copy_buffer(self, dst_buffer, &[copy_region]);
    }
}

impl<L: MemLocation> Drop for Buffer<L> {
    fn drop(&mut self) {
        // Drop buffer.
        unsafe {
            self.loader.destroy_buffer(self.handle, None);
        }
        unsafe { gpu_alloc::free_or_log_on_fail(&self.alloc, &mut self.allocation) };
    }
}

impl Buffer<GpuOnly> {
    pub fn copy_via_staging_buffer(
        &mut self,
        device: &Device,
        cmd_buf: &mut RecordingCmdBuf<impl QueueType>,
        src_data: &[u8],
    ) -> MemResult<()> {
        let mut staging_buffer =
            Buffer::<CpuToGpu>::new_for_slice("Staging buffer", &device, src_data, vk::BufferUsageFlags::TRANSFER_SRC)?;
        staging_buffer.copy_into_buffer(src_data, 0)?;
        staging_buffer.copy_to_buffer(cmd_buf, self, staging_buffer.size());
        Ok(())
    }
}

impl Buffer<CpuToGpu> {
    pub fn copy_into_buffer<T: bytemuck::Pod>(&mut self, data: &[T], offset: usize) -> MemResult<()> {
        // CPU to GPU memory is always mappable.
        let mut memory = self.allocation.try_as_mapped_slab().unwrap();
        presser::copy_from_slice_to_offset_with_align(data, &mut memory, offset, align_of::<T>())?;
        Ok(())
    }
}
