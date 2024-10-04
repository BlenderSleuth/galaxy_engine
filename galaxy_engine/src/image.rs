use std::slice;
use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use crate::command_buffer::CommandBuffer;
use crate::device::{Device, QueueFamily};
use crate::engine::MemResult;
use crate::utils::drop_fail;

//pub struct ImageView<'a> {
//    handle: vk::ImageView,
//    image: &'a Image,
//}

pub struct Image {
    handle: vk::Image,
    allocation: Option<Allocation>,
}

impl Image {
    pub fn new(device: &Device, info: &vk::ImageCreateInfo) -> MemResult<Self> {
        let handle = unsafe { device.device().create_image(info, None) }?;
        // Allocate memory for texture image.
        let texture_memory_requirements = unsafe { device.device().get_image_memory_requirements(handle) };

        let alloc_desc = AllocationCreateDesc {
            name: "Texture Image",
            requirements: texture_memory_requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = device.allocate_memory(&alloc_desc)?;
        unsafe { device.device().bind_image_memory(handle, allocation.memory(), 0) }?;
        Ok(Self { handle, allocation: Some(allocation) })
    }

    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    //pub fn view(&self) -> ImageView {
    //    ImageView {
    //        handle: vk::ImageView::null(),
    //        image: self,
    //    };
    //    todo!();
    //}

    pub fn transition_layout(&self, device: &Device, cmd_pool: vk::CommandPool, _format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> VkResult<()> {
        let cmd_buffer = CommandBuffer::begin_one_time(device, cmd_pool)?;

        let mut image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.handle)
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .layer_count(1)
                .base_array_layer(0)
                .level_count(1)
            );

        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            image_barrier.src_access_mask = vk::AccessFlags2KHR::empty(); // Not waiting on any access.
            image_barrier.dst_access_mask = vk::AccessFlags2KHR::TRANSFER_WRITE;

            image_barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE; // Earliest possible stage.
            image_barrier.dst_stage_mask = vk::PipelineStageFlags2::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            image_barrier.src_access_mask = vk::AccessFlags2KHR::TRANSFER_WRITE; // Wait for transfer to finish.
            image_barrier.dst_access_mask = vk::AccessFlags2KHR::SHADER_READ; // Required for fragment shader.

            image_barrier.src_stage_mask = vk::PipelineStageFlags2::TRANSFER;
            image_barrier.dst_stage_mask = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition.");
        }

        let dependency_info = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .image_memory_barriers(slice::from_ref(&image_barrier));

        unsafe { device.ext().sync2.cmd_pipeline_barrier2(cmd_buffer.handle(), &dependency_info) };

        cmd_buffer.end_and_submit(device.get_queue(QueueFamily::Graphics))
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.device().destroy_image(self.handle, None) };
        if let Some(allocation) = self.allocation.take() {
            drop_fail(device.free_memory(allocation), "Failed to free image memory");
        }
    }
}
