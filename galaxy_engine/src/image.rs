use std::slice;

use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::command_buffer::CommandBuffer;
use crate::device::{Device, QueueFamily};
use crate::engine::MemResult;
use crate::utils;
use crate::utils::drop_fail;

#[ouroboros::self_referencing]
pub struct ImageWithView {
    pub image: Image,
    #[borrows(image)]
    #[covariant]
    pub view: ImageView<'this>,
}

impl ImageWithView {
    pub fn from_image(device: &Device, image: Image, aspect: vk::ImageAspectFlags) -> VkResult<Self> {
         ImageWithViewTryBuilder {
            image,
            view_builder: |image| ImageView::new(device, image, aspect),
        }.try_build()
    }

    pub unsafe fn destroy(mut self, device: &Device) {
        self.with_view_mut(|view| view.destroy(&device));
        let mut heads = self.into_heads();
        heads.image.destroy(device);
    }
}

pub enum ImageRef<'a> {
    External(vk::Image),
    Image(&'a Image),
}

pub struct ImageView<'a> {
    handle: vk::ImageView,
    image: ImageRef<'a>,
}

impl<'a> ImageView<'a> {
    fn view_type_from_image_type(image_type: vk::ImageType, array_layers: u32) -> vk::ImageViewType {
        // Currently doesn't support cube maps.
        match (image_type, array_layers) {
            (vk::ImageType::TYPE_1D, 1) => vk::ImageViewType::TYPE_1D,
            (vk::ImageType::TYPE_1D, 2..) => vk::ImageViewType::TYPE_1D_ARRAY,
            (vk::ImageType::TYPE_2D, 1) => vk::ImageViewType::TYPE_2D,
            (vk::ImageType::TYPE_2D, 2..) => vk::ImageViewType::TYPE_2D_ARRAY,
            (vk::ImageType::TYPE_3D, 1) => vk::ImageViewType::TYPE_3D,
            _ => panic!("Unsupported image type."),
        }
    }

    pub fn new(device: &Device, image: &'a Image, aspect_mask: vk::ImageAspectFlags) -> VkResult<Self> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.handle())
            .view_type(Self::view_type_from_image_type(image.dimensions, image.array_layers))
            .format(image.format)
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .base_mip_level(0)
                .level_count(image.mip_levels)
                .base_array_layer(0)
                .layer_count(image.array_layers)
            );
        let handle = unsafe { device.device().create_image_view(&view_info, None) }?;
        Ok(Self { handle, image: ImageRef::Image(image) })
    }
    pub fn new_external(image: vk::Image, image_view: vk::ImageView) -> VkResult<Self> {
        Ok(Self { handle: image_view, image: ImageRef::External(image) })
    }
    pub fn handle(&self) -> &vk::ImageView {
        &self.handle
    }
    pub unsafe fn destroy(&self, device: &Device) {
        unsafe { device.device().destroy_image_view(self.handle, None) };
    }
}

pub struct Image {
    handle: vk::Image,
    allocation: Option<Allocation>,
    format: vk::Format,
    tiling: vk::ImageTiling,
    dimensions: vk::ImageType,
    extent: vk::Extent3D,
    mip_levels: u32,
    array_layers: u32,
}

impl Image {
    pub fn new(device: &Device, info: &vk::ImageCreateInfo) -> MemResult<Self> {
        let handle = unsafe { device.device().create_image(info, None) }?;

        // Allocate memory for image.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default()
            .push_next(&mut dedicated_requirements);
        let requirements_info = vk::ImageMemoryRequirementsInfo2::default()
            .image(handle);
        unsafe { device.device().get_image_memory_requirements2(&requirements_info, &mut requirements) };

        let requirements = requirements.memory_requirements;
        let allocation_scheme = if utils::use_dedicated_allocation(dedicated_requirements) {
            AllocationScheme::DedicatedImage(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let alloc_desc = AllocationCreateDesc {
            name: "Texture Image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme,
        };
        let allocation = device.allocate_memory(&alloc_desc)?;
        unsafe { device.device().bind_image_memory(handle, allocation.memory(), 0) }?;

        Ok(Self {
            handle,
            allocation: Some(allocation),
            format: info.format,
            tiling: info.tiling,
            dimensions: info.image_type,
            extent: info.extent,
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
        })
    }

    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    pub fn transition_layout(&self, device: &Device, cmd_pool: vk::CommandPool, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, aspect: vk::ImageAspectFlags, src_queue: Option<QueueFamily>, dst_queue: Option<QueueFamily>) -> VkResult<()> {
        let cmd_buffer = CommandBuffer::begin_one_time(device, cmd_pool)?;

        let mut image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue.map(|q| device.get_queue_family_idx(q)).unwrap_or(vk::QUEUE_FAMILY_IGNORED))
            .dst_queue_family_index(dst_queue.map(|q| device.get_queue_family_idx(q)).unwrap_or(vk::QUEUE_FAMILY_IGNORED))
            .image(self.handle)
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
            );

        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            image_barrier.src_access_mask = vk::AccessFlags2KHR::empty(); // Not waiting on any access.
            image_barrier.dst_access_mask = vk::AccessFlags2KHR::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2KHR::DEPTH_STENCIL_ATTACHMENT_WRITE;

            image_barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE; // Earliest possible stage.
            image_barrier.dst_stage_mask = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        } else if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
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

        cmd_buffer.end_submit_and_wait(device.get_queue(QueueFamily::Graphics))
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.device().destroy_image(self.handle, None) };
        if let Some(allocation) = self.allocation.take() {
            drop_fail(device.free_memory(allocation), "Failed to free image memory");
        }
    }
}
