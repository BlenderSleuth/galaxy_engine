// Copyright (c) 2024. Ben Sutherland

use std::mem::ManuallyDrop;
use std::slice;

use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::utils;
use crate::vulkan::buffer::{Buffer, CpuToGpu};
use crate::vulkan::command_buffer::{CommandBuffer, TransientOrPersistentCommandBuffer};
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::vulkan::{debug, gpu_alloc};

pub struct ImageView {
    loader: SharedDeviceLoader,
    handle: vk::ImageView,
}

impl ImageView {
    // Image in view_info must outlive the image view. Usually call Image::get_or_create_view() instead.
    pub unsafe fn new(loader: SharedDeviceLoader, view_info: &vk::ImageViewCreateInfo) -> VkResult<Self> {
        let handle = unsafe { loader.create_image_view(view_info, None) }?;
        Ok(Self { loader, handle })
    }

    pub fn handle(&self) -> vk::ImageView {
        self.handle
    }

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
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_image_view(self.handle, None) };
    }
}

#[allow(dead_code)] // TODO
#[derive(Clone, Copy)]
pub enum ImageDimensions {
    Type1D(u32),
    Type2D(vk::Extent2D),
    Type3D(vk::Extent3D),
}

impl ImageDimensions {
    pub fn num_dimensions(&self) -> u32 {
        match self {
            ImageDimensions::Type1D(_) => 1,
            ImageDimensions::Type2D(_) => 2,
            ImageDimensions::Type3D(_) => 3,
        }
    }
    pub fn image_type(&self) -> vk::ImageType {
        match self {
            ImageDimensions::Type1D(_) => vk::ImageType::TYPE_1D,
            ImageDimensions::Type2D(_) => vk::ImageType::TYPE_2D,
            ImageDimensions::Type3D(_) => vk::ImageType::TYPE_3D,
        }
    }
    pub fn extent(&self) -> vk::Extent3D {
        match self {
            ImageDimensions::Type1D(width) => vk::Extent3D {
                width: *width,
                height: 1,
                depth: 1,
            },
            ImageDimensions::Type2D(extent) => vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            ImageDimensions::Type3D(extent) => *extent,
        }
    }
}

pub struct Image {
    loader: SharedDeviceLoader,
    alloc: SharedAllocator,
    handle: vk::Image,
    allocation: ManuallyFreeAllocation,
    view: ManuallyDrop<ImageView>,
    // format: vk::Format,
    // tiling: vk::ImageTiling,
    // num_dimensions: vk::ImageType,
    extent: vk::Extent3D,
    subresource: vk::ImageSubresourceRange,
}

impl Image {
    // New with an image create info.
    pub fn new(
        name: &str,
        device: &Device,
        info: &vk::ImageCreateInfo,
        subresource: vk::ImageSubresourceRange,
    ) -> MemResult<Self> {
        let handle = unsafe { device.loader().create_image(&info, None) }?;

        // Debug name image object.
        debug::set_object_name(device, handle, name)?;

        // Allocate memory for image.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        let requirements_info = vk::ImageMemoryRequirementsInfo2::default().image(handle);
        unsafe {
            device
                .loader()
                .get_image_memory_requirements2(&requirements_info, &mut requirements)
        };

        let requirements = requirements.memory_requirements;
        let allocation_scheme = if gpu_alloc::use_dedicated_allocation(dedicated_requirements) {
            AllocationScheme::DedicatedImage(handle)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let alloc_desc = AllocationCreateDesc {
            name: debug::debug_only_name!(name),
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme,
        };
        let allocation = device.allocate_and_bind_memory(&alloc_desc, handle)?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(handle)
            .view_type(ImageView::view_type_from_image_type(
                info.image_type,
                subresource.layer_count,
            ))
            .format(info.format)
            .subresource_range(subresource);
        let view = unsafe { ImageView::new(device.cloned_loader(), &view_info)? };

        Ok(Self {
            loader: device.cloned_loader(),
            alloc: device.cloned_allocator(),
            handle,
            allocation,
            view: ManuallyDrop::new(view),
            // format: info.format,
            // tiling: info.tiling,
            // num_dimensions: info.image_type,
            extent: info.extent,
            subresource,
        })
    }

    pub fn new_from_mip_levels(
        name: &str,
        device: &Device,
        gfx_cmd_pool: vk::CommandPool,
        levels: &[&[u8]],
        dimensions: ImageDimensions,
        format: vk::Format,
    ) -> MemResult<Self> {
        let num_mips = levels.len() as u32;
        let total_mip_size: u32 = levels
            .iter()
            .fold(0, |acc, level| acc + level.len())
            .try_into()
            .unwrap();
        let extent = dimensions.extent();
        let image_info = vk::ImageCreateInfo::default()
            .image_type(dimensions.image_type())
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
        let mut image = Image::new(name, device, &image_info, subresource)?;

        let mut image_buffer = Buffer::<CpuToGpu>::new(
            debug::debug_only_name!("{name} staging buffer"),
            &device,
            total_mip_size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        // Copy mip levels into buffer.
        let mut regions = Vec::with_capacity(levels.len());
        let mut offset = 0;
        for (mip_level, data) in levels.iter().enumerate() {
            let mip_level = mip_level as u32;
            regions.push(
                vk::BufferImageCopy::default()
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
                        height: if dimensions.num_dimensions() >= 2 {
                            extent.height >> mip_level
                        } else {
                            1
                        },
                        depth: if dimensions.num_dimensions() >= 3 {
                            extent.depth >> mip_level
                        } else {
                            1
                        },
                    }),
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
        )?;

        // Perform the copy.
        unsafe {
            device.loader().cmd_copy_buffer_to_image(
                cmd_buffer.handle(),
                image_buffer.handle(),
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            )
        };

        // Transition all mip levels to shader read only optimal.
        image.transition_layout(
            &device,
            cmd_buffer.as_persistent(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            None,
        )?;

        cmd_buffer.end_submit_and_wait(device, device.primary_queue().handle())?;

        Ok(image)
    }

    //pub fn handle(&self) -> vk::Image {
    //    self.handle
    //}

    pub fn view(&self) -> &ImageView {
        &self.view
    }

    pub fn transition_layout(
        &mut self,
        device: &Device,
        cmd: TransientOrPersistentCommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_level: Option<u32>,
    ) -> VkResult<()> {
        let cmd_buffer = cmd.command_buffer();

        let mut image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
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
            image_barrier.dst_access_mask = vk::AccessFlags2KHR::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2KHR::DEPTH_STENCIL_ATTACHMENT_WRITE;

            image_barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE; // Earliest possible stage.
            image_barrier.dst_stage_mask = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        } else if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            image_barrier.src_access_mask = vk::AccessFlags2KHR::empty(); // Not waiting on any access.
            image_barrier.dst_access_mask = vk::AccessFlags2KHR::TRANSFER_WRITE;

            image_barrier.src_stage_mask = vk::PipelineStageFlags2::TOP_OF_PIPE; // Earliest possible stage.
            image_barrier.dst_stage_mask = vk::PipelineStageFlags2::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
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

        unsafe {
            device
                .extensions()
                .sync2
                .cmd_pipeline_barrier2(cmd_buffer.handle(), &dependency_info)
        };

        cmd.maybe_end_submit_and_wait(device, device.primary_queue().handle())
    }

    #[allow(dead_code)] // TODO
    pub fn copy_buffer_to_image(
        &mut self,
        device: &Device,
        cmd: TransientOrPersistentCommandBuffer,
        buffer: &Buffer<CpuToGpu>,
        mip_level: u32,
    ) -> VkResult<()> {
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

        unsafe {
            device.loader().cmd_copy_buffer_to_image(
                cmd_buffer.handle(),
                buffer.handle(),
                self.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        };

        cmd.maybe_end_submit_and_wait(device, device.primary_queue().handle())
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // Drop image view.
        unsafe { ManuallyDrop::drop(&mut self.view) };
        // Drop image.
        unsafe { self.loader.destroy_image(self.handle, None) };
        // Free memory.
        unsafe { gpu_alloc::free_or_log_on_fail(&self.alloc, &mut self.allocation) };
    }
}
