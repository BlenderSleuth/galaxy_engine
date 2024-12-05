// Copyright (c) 2024 Ben Sutherland.

use ash::vk;

use crate::engine::GalaxyEngine;
use crate::vulkan::buffer::{Buffer, CpuToGpu};
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;
// How often is this resource updated?
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum ResourceFrequency {
//     MultiFrame,
//     SingleFrame,
//     DrawCall,
// }
//
// pub trait ResourceFrequencyTrait {
//     fn frequency() -> ResourceFrequency;
// }
// pub struct MultiFrame;
// impl ResourceFrequencyTrait for MultiFrame {
//     fn frequency() -> ResourceFrequency { ResourceFrequency::MultiFrame }
// }
// pub struct SingleFrame;
// impl ResourceFrequencyTrait for SingleFrame {
//     fn frequency() -> ResourceFrequency { ResourceFrequency::SingleFrame }
// }
// pub struct DrawCall;
// impl ResourceFrequencyTrait for DrawCall {
//     fn frequency() -> ResourceFrequency { ResourceFrequency::DrawCall }
// }

pub enum VolatileBufferType {
    Uniform,
    Storage,
}

impl VolatileBufferType {
    pub fn usage(&self) -> vk::BufferUsageFlags {
        match self {
            Self::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            Self::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }
    pub fn min_align(&self, device: &Device) -> usize {
        let limits = &device.physical_device().properties.base.limits;
        match self {
            Self::Uniform => limits.min_uniform_buffer_offset_alignment as usize,
            Self::Storage => limits.min_storage_buffer_offset_alignment as usize,
        }
    }
}

// A multi-buffered uniform buffer that can be updated every frame.
// Update the local data and call copy_to_gpu to update the GPU buffer.
pub struct VolatileBuffer<T: bytemuck::Pod, const N: usize = { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }> {
    buffer: Buffer<CpuToGpu>,
    size: usize,
    marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod, const N: usize> VolatileBuffer<T, N> {
    pub fn new(name: &str, device: &Device, buffer_type: VolatileBufferType) -> MemResult<Self> {
        // Calculate the padded size of the buffer, based on the alignment requirements of the buffer type.
        let size = core::alloc::Layout::new::<T>()
            .align_to(buffer_type.min_align(device))
            .unwrap()
            .pad_to_align()
            .size();

        let mut buffer = Buffer::new(
            name,
            device,
            (size * N) as vk::DeviceSize,
            buffer_type.usage(),
            Some(device.physical_device().volatile_memory_type.type_bits),
        )?;

        // Zero-init memory (which allows it to be soundly casted to a Pod type).
        buffer.zero_memory();

        Ok(Self {
            buffer,
            size,
            marker: std::marker::PhantomData,
        })
    }

    fn frame_offset(&self, frame: usize) -> usize {
        debug_assert!(frame < N);
        self.size * frame
    }

    //fn frame_range(&self, frame: usize) -> std::ops::Range<usize> {
    //    let offset = self.frame_offset(frame) as usize;
    //    offset..(offset + self.size)
    //}

    //pub fn copy_to_gpu(&mut self, frame_index: usize) -> MemResult<()> {
    //    self.buffer
    //        .copy_into_buffer(&self.local, self.frame_offset(frame_index) as usize)
    //}

    pub fn get_mut(&mut self, frame: usize) -> &mut T {
        // Safety: The buffer is zero-initialized, so this is a safe operation.
        unsafe { self.buffer.get_mut(self.frame_offset(frame)) }
    }

    pub fn descriptor_buffer_info(&self, frame: usize) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer.handle())
            .offset(self.frame_offset(frame) as vk::DeviceSize)
            .range(std::mem::size_of::<T>() as vk::DeviceSize)
    }
}

//impl<T: bytemuck::Pod, const N: usize> VolatileBuffer<[T; N]> {
//    pub fn iter(&self) -> impl Iterator<Item = &T> {
//        self.local.iter()
//    }
//
//    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
//        self.local.iter_mut()
//    }
//}
