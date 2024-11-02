// Copyright (c) 2024 Ben Sutherland.

use std::mem::ManuallyDrop;
use std::slice;

use ash::prelude::VkResult;
use ash::{khr, vk};

use crate::utils;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::image::{Image, ImageView};
use crate::vulkan::instance::Instance;
use crate::vulkan::queue::queue_type::PrimaryQueue;
use crate::vulkan::queue::Queue;
use crate::vulkan::surface::Surface;

pub struct Swapchain {
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<ImageView>,
    colour_resolve_image: ManuallyDrop<Image>,
    depth_image: ManuallyDrop<Image>,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
        surface: &Surface,
        window_size: vk::Extent2D,
        old_swapchain: Option<&Swapchain>,
    ) -> MemResult<Self> {
        let device_properties = device.physical_device();

        // Create swapchain.

        let surface_capabilities = surface.get_capabilities(device.physical_device().handle)?;

        // Choose swap extent.
        let swapchain_extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_size.width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: window_size.height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            }
        };
        let swapchain_format = device_properties.swapchain_format;

        let loader = khr::swapchain::Device::new(instance.loader(), &device.loader());
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(device_properties.swapchain_image_count)
            .image_format(swapchain_format.format)
            .image_color_space(swapchain_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(device_properties.presentation_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.map_or(vk::SwapchainKHR::null(), |swapchain| swapchain.handle));

        let handle = unsafe { loader.create_swapchain(&swapchain_info, None) }?;

        // Get swapchain images.
        let images = unsafe { loader.get_swapchain_images(handle) }?;

        // Create image views.
        let mut image_view_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_format.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(utils::DEFAULT_SUBRESOURCE_RANGE);
        let image_views = images
            .iter()
            .map(|swapchain_image| {
                image_view_info.image = *swapchain_image;
                unsafe { ImageView::new(device.cloned_loader(), &image_view_info) }
            })
            .collect::<VkResult<Vec<_>>>()?;

        let msaa_samples = device.physical_device().max_msaa_samples;

        // Create colour resolve image.
        let colour_resolve_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(swapchain_extent.into())
            .mip_levels(1)
            .array_layers(1)
            .format(device_properties.swapchain_format.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);
        let colour_resolve_image = Image::new(
            "Colour resolve image",
            &device,
            &colour_resolve_info,
            Self::get_subresource_range(),
        )?;

        // Create depth image.
        let depth_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(device.physical_device().depth_stencil_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);
        let depth_subresource = vk::ImageSubresourceRange {
            aspect_mask: utils::get_aspect_for_format(device.physical_device().depth_stencil_format),
            ..Self::get_subresource_range()
        };

        let mut depth_image = Image::new("Depth image", device, &depth_image_info, depth_subresource)?;
        let mut cmd_buffer = cmd_pool.allocate_transient_cmd_buffer()?;
        depth_image.transition_layout(
            device.extensions(),
            &mut cmd_buffer,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            None,
        );
        cmd_buffer.end_submit_wait_and_free()?;

        Ok(Self {
            loader,
            handle,
            images,
            image_views,
            colour_resolve_image: ManuallyDrop::new(colour_resolve_image),
            depth_image: ManuallyDrop::new(depth_image),
            extent: swapchain_extent,
            msaa_samples,
        })
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn get_colour_resolve_view(&self) -> &Image {
        &self.colour_resolve_image
    }

    pub fn get_images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn get_image_views(&self) -> &[ImageView] {
        &self.image_views
    }

    pub fn get_depth_view(&self) -> &ImageView {
        self.depth_image.view()
    }

    pub fn get_subresource_range() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }

    pub fn samples(&self) -> vk::SampleCountFlags {
        self.msaa_samples
    }

    pub fn acquire_next_image(&self, semaphore: vk::Semaphore, fence: vk::Fence) -> VkResult<(u32, bool)> {
        unsafe { self.loader.acquire_next_image(self.handle, u64::MAX, semaphore, fence) }
    }

    pub fn queue_present(
        &self,
        queue: &mut Queue<PrimaryQueue>,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> VkResult<bool> {
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(slice::from_ref(&self.handle))
            .image_indices(slice::from_ref(&image_index));
        unsafe { self.loader.queue_present(queue.handle(), &present_info) }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // Drop depth image and view.
        unsafe { ManuallyDrop::drop(&mut self.depth_image) };

        // Drop colour resolve image.
        unsafe { ManuallyDrop::drop(&mut self.colour_resolve_image) };

        // Drop image views.
        self.image_views.clear();
        self.images.clear();

        // Drop swapchain (also drops images).
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}
