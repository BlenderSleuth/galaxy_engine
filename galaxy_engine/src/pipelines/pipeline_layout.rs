// Copyright (c) 2024 Ben Sutherland.

use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::vulkan::device::{Device, SharedDeviceLoader};

pub struct PipelineLayout {
    loader: SharedDeviceLoader,
    handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn new(
        device: &Device,
        descriptor_set_layout: Option<&[vk::DescriptorSetLayout]>,
        push_constant_range: Option<&vk::PushConstantRange>,
    ) -> VkResult<Self> {
        let loader = device.cloned_loader();

        let mut pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        if let Some(descriptor_set_layout) = descriptor_set_layout {
            pipeline_layout_info = pipeline_layout_info.set_layouts(descriptor_set_layout);
        }
        if let Some(push_constant_range) = push_constant_range {
            pipeline_layout_info = pipeline_layout_info.push_constant_ranges(slice::from_ref(&push_constant_range));
        }
        let handle = unsafe { loader.create_pipeline_layout(&pipeline_layout_info, None) }?;

        Ok(Self { loader, handle })
    }
    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_pipeline_layout(self.handle, None) }
    }
}
