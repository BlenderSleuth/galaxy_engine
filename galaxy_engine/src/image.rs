use std::slice;

use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::buffer::Buffer;
use crate::buffer::mem_location::{CpuToGpu};
use crate::command_buffer::{CommandBuffer, TransientOrPersistentCommandBuffer};
use crate::device::{Device, QueueFamily};
use crate::engine::MemResult;
use crate::utils;

#[ouroboros::self_referencing]
pub struct ImageWithView {
    pub image: Image,
    #[borrows(image)]
    #[covariant]
    pub view: ImageView<'this>,
}

impl ImageWithView {
    pub fn from_image(device: &Device, image: Image) -> VkResult<Self> {
        ImageWithViewTryBuilder {
            image,
            view_builder: |image| ImageView::new(device, image),
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

    pub fn new(device: &Device, image: &'a Image) -> VkResult<Self> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.handle())
            .view_type(Self::view_type_from_image_type(image.num_dimensions, image.subresource.layer_count))
            .format(image.format)
            .subresource_range(image.subresource);
        let handle = unsafe { device.device().create_image_view(&view_info, None) }?;
        Ok(Self { handle, image: ImageRef::Image(image) })
    }
    pub fn new_external(device: &Device, image: vk::Image, view_info: &vk::ImageViewCreateInfo) -> VkResult<Self> {
        let handle = unsafe { device.device().create_image_view(view_info, None) }?;
        Ok(Self { handle, image: ImageRef::External(image) })
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
    num_dimensions: vk::ImageType,
    extent: vk::Extent3D,
    subresource: vk::ImageSubresourceRange,
}

impl Image {
    // New with an image create info.
    pub fn new(device: &Device, info: &vk::ImageCreateInfo, subresource: vk::ImageSubresourceRange, name: &str) -> MemResult<Self> {
        let handle = unsafe { device.device().create_image(&info, None) }?;

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
            name,
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
            num_dimensions: info.image_type,
            extent: info.extent,
            subresource,
        })
    }

    pub fn new_from_mip_levels(
        device: &Device,
        gfx_cmd_pool: vk::CommandPool,
        levels: &[&[u8]],
        num_dimensions: vk::ImageType,
        extent: vk::Extent3D,
        format: vk::Format,
        name: &str,
    ) -> MemResult<Self> {
        let num_mips = levels.len() as u32;
        let total_mip_size: u32 = levels.iter().fold(0, |acc, level| acc + level.len()).try_into().unwrap();
        let image_info = vk::ImageCreateInfo::default()
            .image_type(num_dimensions)
            .extent(extent)
            .mip_levels(num_mips)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        let subresource = vk::ImageSubresourceRange {
            aspect_mask: utils::get_aspect_for_format(format),
            base_mip_level: 0,
            level_count: num_mips,
            ..utils::DEFAULT_SUBRESOURCE_RANGE
        };
        let mut image = Image::new(device, &image_info, subresource, name)?;

        let mut image_buffer = Buffer::<CpuToGpu>::new(
            &device,
            &format!("{name} staging buffer"), // TODO: Resource names only in debug.
            total_mip_size,
            std::mem::size_of::<u8>(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;

        // Copy mip levels into buffer.
        let mut regions = Vec::with_capacity(levels.len());
        let mut offset = 0;
        for (mip_level, data) in levels.iter().enumerate() {
            let mip_level = mip_level as u32;
            regions.push(vk::BufferImageCopy::default()
                .buffer_offset(offset as vk::DeviceSize)
                // Tight packed data.
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: subresource.aspect_mask,
                    mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: extent.width >> mip_level,
                    height: if num_dimensions >= vk::ImageType::TYPE_2D { extent.height >> mip_level} else { 1 },
                    depth: if num_dimensions >= vk::ImageType::TYPE_3D { extent.depth >> mip_level} else { 1 },
                })
            );
            image_buffer.copy_into_buffer(&data, offset)?;
            offset += data.len();
        }

        let cmd_buffer = CommandBuffer::begin_one_time(device, gfx_cmd_pool)?;

        // Transition all mip levels to transfer destination optimal.
        image.transition_layout(
            &device,
            cmd_buffer.as_persistent(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            None,
            None,
            None,
        )?;

        // Perform the copy.
        unsafe { device.device().cmd_copy_buffer_to_image(cmd_buffer.handle(), image_buffer.handle(), image.handle, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions) };

        // Transition all mip levels to shader read only optimal.
        image.transition_layout(
            &device,
            cmd_buffer.as_persistent(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            None,
            None,
            None,
        )?;

        cmd_buffer.end_submit_and_wait(device, device.get_queue(QueueFamily::Graphics))?;

        unsafe { image_buffer.destroy(&device) }?;

        Ok(image)
    }

    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    pub fn transition_layout(&mut self,
                             device: &Device,
                             cmd: TransientOrPersistentCommandBuffer,
                             old_layout: vk::ImageLayout,
                             new_layout: vk::ImageLayout,
                             mip_level: Option<u32>,
                             src_queue: Option<QueueFamily>,
                             dst_queue: Option<QueueFamily>,
    ) -> VkResult<()> {
        let cmd_buffer = cmd.command_buffer();

        let mut image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue.map(|q| device.get_queue_family_idx(q)).unwrap_or(vk::QUEUE_FAMILY_IGNORED))
            .dst_queue_family_index(dst_queue.map(|q| device.get_queue_family_idx(q)).unwrap_or(vk::QUEUE_FAMILY_IGNORED))
            .image(self.handle)
            .subresource_range(if let Some(mip_level) = mip_level {
                // If a mip level is specified, only transition that level.
                vk::ImageSubresourceRange {
                    base_mip_level: mip_level,
                    level_count: 1,
                    ..self.subresource
                }
            } else {
                self.subresource
            });

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

        cmd.maybe_end_submit_and_wait(device, device.get_queue(QueueFamily::Graphics))
    }

    pub fn copy_buffer_to_image(&mut self, device: &Device, cmd: TransientOrPersistentCommandBuffer, buffer: &Buffer<CpuToGpu>, mip_level: u32, queue: QueueFamily) -> VkResult<()> {
        let cmd_buffer = cmd.command_buffer();

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: self.subresource.aspect_mask,
                mip_level,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(self.extent);

        unsafe { device.device().cmd_copy_buffer_to_image(cmd_buffer.handle(), buffer.handle(), self.handle, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]) };

        cmd.maybe_end_submit_and_wait(device, device.get_queue(queue))
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.device().destroy_image(self.handle, None) };
        if let Some(allocation) = self.allocation.take() {
            utils::drop_fail(device.free_memory(allocation), "Failed to free image memory");
        }
    }
}
