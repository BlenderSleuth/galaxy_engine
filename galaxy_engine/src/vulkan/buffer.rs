// Copyright (c) 2024 Ben Sutherland.

use std::mem::MaybeUninit;
use std::num::NonZeroU32;
use std::ops::DerefMut;

use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use presser::Slab;

use crate::utils::ScopeGuard;
use crate::vulkan::command_buffer::RecordingCmdBuf;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::vulkan::queue::queue_type::QueueType;
use crate::vulkan::{debug, gpu_alloc};

// Type-state trait encoding of gpu_allocator::MemoryLocation for use in generic parameters.
pub trait MemLocation {
    fn new(loader: &ash::Device, handle: vk::Buffer) -> Self;
    fn location() -> MemoryLocation;
    fn extra_usage_flags() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::empty()
    }
    fn memory_type_override(_device: &Device) -> Option<NonZeroU32> {
        None
    }
}

pub struct GpuOnly {
    device_addr: vk::DeviceAddress,
}
impl MemLocation for GpuOnly {
    fn new(loader: &ash::Device, handle: vk::Buffer) -> Self {
        let info = vk::BufferDeviceAddressInfo::default().buffer(handle);
        Self {
            device_addr: unsafe { loader.get_buffer_device_address(&info) },
        }
    }
    fn location() -> MemoryLocation {
        MemoryLocation::GpuOnly
    }
    fn extra_usage_flags() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
    }
}

pub trait HostVisible: MemLocation {}

pub struct HostVisibleDeviceLocal;
impl MemLocation for HostVisibleDeviceLocal {
    fn new(_loader: &ash::Device, _handle: vk::Buffer) -> Self {
        Self
    }
    fn location() -> MemoryLocation {
        MemoryLocation::CpuToGpu
    }
    fn memory_type_override(device: &Device) -> Option<NonZeroU32> {
        Some(device.physical_device().volatile_memory_type.type_bits)
    }
}
impl HostVisible for HostVisibleDeviceLocal {}

pub struct Staging;
impl MemLocation for Staging {
    fn new(_loader: &ash::Device, _handle: vk::Buffer) -> Self {
        Self
    }
    fn location() -> MemoryLocation {
        MemoryLocation::CpuToGpu
    }
    fn memory_type_override(device: &Device) -> Option<NonZeroU32> {
        Some(device.physical_device().staging_memory_type.type_bits)
    }
}
impl HostVisible for Staging {}

pub struct GpuToCpu;
impl MemLocation for GpuToCpu {
    fn new(_loader: &ash::Device, _handle: vk::Buffer) -> Self {
        Self
    }
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
    mem_location: L,
}

impl<L: MemLocation> Buffer<L> {
    pub fn new_for_type<T: bytemuck::Pod>(name: &str, device: &Device, usage: vk::BufferUsageFlags) -> MemResult<Self> {
        Self::new(name, device, size_of::<T>() as vk::DeviceSize, usage)
    }

    pub fn new_for_slice<T: bytemuck::Pod>(
        name: &str,
        device: &Device,
        slice: &[T],
        usage: vk::BufferUsageFlags,
    ) -> MemResult<Self> {
        Self::new(name, device, size_of_val(slice) as vk::DeviceSize, usage)
    }

    pub fn new(name: &str, device: &Device, size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> MemResult<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | L::extra_usage_flags())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.loader().create_buffer(&buffer_info, None) }?;
        let mut guard = ScopeGuard::new(|| unsafe { device.loader().destroy_buffer(handle, None) });

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

        let mut requirements = requirements.memory_requirements;

        // Allows using a more specific type of memory.
        if let Some(override_memory_type_bits) = L::memory_type_override(device).map(|n| n.get()) {
            let overlap = override_memory_type_bits & requirements.memory_type_bits;
            if overlap == 0 {
                log::warn!("Buffer cannot use override memory type - using required memory type.")
            } else {
                requirements.memory_type_bits = overlap;
            }
        }

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

        guard.defuse();

        Ok(Self {
            loader: device.cloned_loader(),
            alloc: device.cloned_allocator(),
            handle,
            allocation,
            size,
            mem_location: L::new(device.loader(), handle),
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
    pub fn copy_via_staging_buffer_with<Q: QueueType, F: FnOnce(&mut Buffer<Staging>) -> MemResult<()>>(
        &mut self,
        device: &Device,
        cmd_buf: &mut RecordingCmdBuf<Q>,
        size: Option<vk::DeviceSize>,
        f: F,
    ) -> MemResult<Buffer<Staging>> {
        let mut staging_buffer = Buffer::<Staging>::new(
            "Staging buffer",
            device,
            size.unwrap_or(self.size),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        f(&mut staging_buffer)?;
        staging_buffer.copy_to_buffer(cmd_buf, self, staging_buffer.size());
        Ok(staging_buffer)
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.mem_location.device_addr
    }
}

impl<L: HostVisible> Buffer<L> {
    fn get_mapped_memory(&mut self) -> &mut impl Slab {
        // Host visible memory is always mappable.
        self.allocation.deref_mut()
    }

    pub fn zero_memory(&mut self) {
        self.allocation.as_maybe_uninit_bytes_mut().fill(MaybeUninit::zeroed());
    }

    /// # Safety
    ///
    /// Buffer memory must be initialised.
    pub unsafe fn get_mut_bytes(&mut self, size: usize, offset: usize) -> &mut [u8] {
        let range = offset..(offset + size);
        unsafe { self.allocation.assume_range_initialized_as_bytes_mut(range) }
    }

    /// Casts the buffer memory to a mutable reference of type T.
    /// # Safety
    ///
    /// Buffer memory must be initialised.
    pub unsafe fn get_mut<T: bytemuck::Pod>(&mut self, offset: usize) -> &mut T {
        let size = std::mem::size_of::<T>();
        // Panics if the alignment is wrong.
        bytemuck::try_from_bytes_mut(self.get_mut_bytes(size, offset)).unwrap()
    }

    /// Casts the buffer memory to a mutable slice of type T.
    /// # Safety
    ///
    /// Buffer memory must be initialised.
    pub unsafe fn get_mut_slice<T: bytemuck::Pod>(&mut self, len: usize, offset: usize) -> &mut [T] {
        let size = std::mem::size_of::<T>() * len;
        // Panics if the alignment is wrong.
        bytemuck::try_cast_slice_mut(self.get_mut_bytes(size, offset)).unwrap()
    }

    pub fn zero_and_get_mut_bytes(&mut self) -> &mut [u8] {
        self.zero_memory();
        unsafe { self.allocation.assume_initialized_as_bytes_mut() }
    }

    pub fn copy_into_buffer<T: bytemuck::Pod>(&mut self, data: &T, offset: usize) -> MemResult<()> {
        presser::copy_to_offset(data, self.get_mapped_memory(), offset)?;
        Ok(())
    }

    pub fn copy_slice_into_buffer<T: bytemuck::Pod>(&mut self, data: &[T], offset: usize) -> MemResult<()> {
        presser::copy_from_slice_to_offset(data, self.get_mapped_memory(), offset)?;
        Ok(())
    }
}
