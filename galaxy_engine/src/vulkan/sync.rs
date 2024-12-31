// Copyright (c) 2024-2025 Ben Sutherland.

use ash::prelude::VkResult;
use ash::vk;

use crate::vulkan::device::Device;
use crate::vulkan::get_device_loader;

pub trait Semaphore {
    fn handle(&self) -> vk::Semaphore;
}

pub struct WaitSemaphore {
    pub handle: vk::Semaphore,
    pub stage_mask: vk::PipelineStageFlags,
}

pub struct BinarySemaphore {
    handle: vk::Semaphore,
}

impl BinarySemaphore {
    pub fn new(device: &Device) -> VkResult<Self> {
        let handle = unsafe { device.loader().create_semaphore(&Default::default(), None) }?;
        Ok(Self { handle })
    }
}

impl Semaphore for BinarySemaphore {
    fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe { get_device_loader().destroy_semaphore(self.handle, None) }
    }
}

//pub struct TimelineSemaphore {
//    loader: SharedDeviceLoader,
//    handle: vk::Semaphore,
//}

pub struct Fence {
    handle: vk::Fence,
}

impl Fence {
    pub fn new(loader: &ash::Device, signaled: bool) -> VkResult<Self> {
        let handle = unsafe {
            loader.create_fence(
                &vk::FenceCreateInfo::default().flags(if signaled {
                    vk::FenceCreateFlags::SIGNALED
                } else {
                    vk::FenceCreateFlags::empty()
                }),
                None,
            )
        }?;
        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn reset(&self, loader: &ash::Device) -> VkResult<()> {
        unsafe { loader.reset_fences(&[self.handle]) }
    }

    pub fn wait_with_timeout(&self, loader: &ash::Device, timeout: u64) -> VkResult<()> {
        unsafe { loader.wait_for_fences(&[self.handle], true, timeout) }
    }

    pub fn wait(&self, loader: &ash::Device) -> VkResult<()> {
        self.wait_with_timeout(loader, u64::MAX)
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { get_device_loader().destroy_fence(self.handle, None) }
    }
}
