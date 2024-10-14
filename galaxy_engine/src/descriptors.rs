// Copyright (c) 2024. Ben Sutherland

use ash::prelude::VkResult;
use ash::vk;

use crate::device::{get_device, Device};

pub struct DescriptorPool {
    handle: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn new(device: &Device, info: &vk::DescriptorPoolCreateInfo) -> VkResult<Self> {
        let handle = unsafe { device.loader().create_descriptor_pool(&info, None) }?;
        Ok(Self { handle })
    }
    pub fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { get_device().destroy_descriptor_pool(self.handle, None) }
    }
}
