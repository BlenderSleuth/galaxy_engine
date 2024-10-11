use crate::device::{get_device, Device};
use ash::prelude::VkResult;
use ash::vk;

pub struct BinarySemaphore {
    handle: vk::Semaphore,
}

impl BinarySemaphore {
    pub fn new(device: &Device) -> VkResult<Self> {
        let handle = unsafe { device.loader().create_semaphore(&Default::default(), None) }?;
        Ok(Self { handle })
    }
    pub fn handle(&self) -> vk::Semaphore { self.handle }
    pub fn ref_handle(&self) -> &vk::Semaphore { &self.handle }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe { get_device().destroy_semaphore(self.handle, None) }
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
    pub fn new(device: &Device, signaled: bool) -> VkResult<Self> {
        let handle = unsafe {
            device.loader().create_fence(
                &vk::FenceCreateInfo::default().flags(if signaled {
                    vk::FenceCreateFlags::SIGNALED
                } else {
                    vk::FenceCreateFlags::empty()
                }), None)
        }?;
        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Fence { self.handle }

    pub fn reset(&self) -> VkResult<()> {
        unsafe { get_device().reset_fences(&[self.handle]) }
    }

    pub fn wait(&self, timeout: u64) -> VkResult<()> {
        unsafe { get_device().wait_for_fences(&[self.handle], true, timeout) }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { get_device().destroy_fence(self.handle, None) }
    }
}