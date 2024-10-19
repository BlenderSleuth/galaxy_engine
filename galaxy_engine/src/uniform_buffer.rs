// Copyright (c) 2024. Ben Sutherland

use ash::vk;

use crate::buffer::{Buffer, CpuToGpu, GpuOnly, MemLocation};
use crate::engine::GalaxyEngine;
use crate::gpu_alloc::MemResult;
use crate::vulkan::{debug, Device};

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

// A multi-buffered uniform buffer that can be updated every frame.
pub struct VolatileUniformBuffer {
    staging_buffer: Buffer<CpuToGpu>,
    gpu_buffer: Buffer<GpuOnly>,
    size: vk::DeviceSize,
}

impl VolatileUniformBuffer {
    const N: usize = GalaxyEngine::MAX_FRAMES_IN_FLIGHT;

    fn create_buffer<L: MemLocation>(
        name: &str,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> MemResult<Buffer<L>> {
        Buffer::new(name, device, size, usage)
    }

    pub fn new_for_type<T: bytemuck::Pod>(name: &str, device: &Device) -> MemResult<Self> {
        Self::new(name, device, std::mem::size_of::<T>() as vk::DeviceSize)
    }

    pub fn new(name: &str, device: &Device, size: vk::DeviceSize) -> MemResult<Self> {
        Ok(Self {
            staging_buffer: Self::create_buffer(
                debug::debug_only_name!("{name} staging"),
                device,
                size * Self::N as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?,
            gpu_buffer: Self::create_buffer(
                debug::debug_only_name!("{name} GPU"),
                device,
                size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?,
            size: size as vk::DeviceSize,
        })
    }

    fn frame_offset(&self, frame: usize) -> vk::DeviceSize {
        debug_assert!(frame < Self::N);
        self.size * frame as vk::DeviceSize
    }

    pub fn update<T: bytemuck::Pod>(&mut self, current_frame: usize, data: &[T]) -> MemResult<()> {
        self.staging_buffer
            .copy_into_buffer(data, self.frame_offset(current_frame) as usize)
    }

    pub fn copy_to_gpu(&self, loader: &ash::Device, current_frame: usize, cmd_buffer: vk::CommandBuffer) {
        let copy_region = vk::BufferCopy::default()
            .src_offset(self.frame_offset(current_frame))
            .size(self.size);

        unsafe {
            loader.cmd_copy_buffer(
                cmd_buffer,
                self.staging_buffer.handle(),
                self.gpu_buffer.handle(),
                &[copy_region],
            )
        };
    }

    //pub fn gpu_buffer_handle(&self) -> vk::Buffer {
    //    self.gpu_buffer.handle()
    //}

    pub fn descriptor_buffer_info(&self) -> vk::DescriptorBufferInfo {
        self.gpu_buffer.descriptor_buffer_info()
    }
}
