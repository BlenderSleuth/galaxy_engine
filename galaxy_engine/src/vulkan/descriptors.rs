// Copyright (c) 2024-2025 Ben Sutherland.

use std::ops::Range;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;

use crate::vulkan::device::{Device, SharedDeviceLoader};

pub struct DescriptorPool<const N: usize> {
    loader: SharedDeviceLoader,
    handle: vk::DescriptorPool,
    sets: ArrayVec<vk::DescriptorSet, N>,
}

impl<const N: usize> DescriptorPool<N> {
    pub fn new(device: &Device, pool_sizes: &[vk::DescriptorPoolSize]) -> VkResult<Self> {
        let info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(pool_sizes)
            .max_sets(N as u32);
        let handle = unsafe { device.loader().create_descriptor_pool(&info, None) }?;
        Ok(Self {
            loader: device.cloned_loader(),
            handle,
            sets: ArrayVec::new(),
        })
    }

    pub fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }

    pub fn iter(&self) -> impl Iterator<Item = &vk::DescriptorSet> {
        self.sets.iter()
    }

    pub fn get(&self, index: usize) -> vk::DescriptorSet {
        self.sets[index]
    }

    // Returns the range of indices of the allocated descriptor sets.
    pub fn allocate_descriptor_sets(
        &mut self,
        device: &Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> VkResult<Range<usize>> {
        assert!(self.sets.len() + layouts.len() <= N as usize);
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle())
            .set_layouts(layouts);
        let start = self.sets.len();
        self.sets
            .extend(unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?);
        Ok(Range {
            start,
            end: self.sets.len(),
        })
    }
}

impl<const N: usize> Drop for DescriptorPool<N> {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_descriptor_pool(self.handle, None) }
    }
}
