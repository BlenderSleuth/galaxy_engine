use std::mem::ManuallyDrop;
use std::slice;
use ash::{khr, vk};
use ash::prelude::VkResult;

use crate::command_buffer::CommandBuffer;
use crate::device::{Device, PhysicalDeviceProperties, PropertyQueueList};
use crate::gpu_alloc::MemResult;
use crate::image::{Image, ImageView};
use crate::surface::Surface;
use crate::utils;

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
        instance: &ash::Instance,
        device: &Device,
        gfx_cmd_pool: vk::CommandPool,
        surface: &Surface,
        window_size: vk::Extent2D,
        old_swapchain: Option<&Swapchain>,
    ) -> MemResult<Self> {
        let device_properties = device.get_properties();
        let unique_queue_families = device_properties.get_unique_queue_families();

        // Create swapchain. TODO: keep on present/graphics queue family only.
        let (image_sharing_mode, queue_family_indices) = if unique_queue_families.len() > 1 {
            (vk::SharingMode::CONCURRENT, unique_queue_families)
        } else {
            (vk::SharingMode::EXCLUSIVE, PropertyQueueList::new())
        };

        let surface_capabilities = surface.get_capabilities(device.get_properties().physical_device)?;

        // Choose swap extent.
        let swapchain_extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_size.width.clamp(surface_capabilities.min_image_extent.width, surface_capabilities.max_image_extent.width),
                height: window_size.height.clamp(surface_capabilities.min_image_extent.height, surface_capabilities.max_image_extent.height),
            }
        };
        let swapchain_format = device_properties.swapchain_format;

        let loader = khr::swapchain::Device::new(&instance, &device.loader());
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(device_properties.image_count)
            .image_format(swapchain_format.format)
            .image_color_space(swapchain_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
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
        let image_views = images.iter().map(|swapchain_image| {
            image_view_info.image = *swapchain_image;
            unsafe { ImageView::new(device.cloned_loader(), &image_view_info) }
        }).collect::<VkResult<Vec<_>>>()?;

        let msaa_samples = PhysicalDeviceProperties::MSAA_SAMPLES;

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
            .extent(vk::Extent3D { width: swapchain_extent.width, height: swapchain_extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(PhysicalDeviceProperties::DEPTH_STENCIL_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);
        let depth_subresource = vk::ImageSubresourceRange {
            aspect_mask: utils::get_aspect_for_format(PhysicalDeviceProperties::DEPTH_STENCIL_FORMAT),
            ..Self::get_subresource_range()
        };

        let mut depth_image = Image::new("Depth image", &device, &depth_image_info, depth_subresource)?;
        depth_image.transition_layout(
            &device,
            CommandBuffer::one_time_transient(&device, gfx_cmd_pool)?,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            None,
            None,
            None,
        )?;

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

    pub fn queue_present(&self, queue: vk::Queue, image_index: u32, wait_semaphores: &[vk::Semaphore]) -> VkResult<bool> {
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(slice::from_ref(&self.handle))
            .image_indices(slice::from_ref(&image_index));
        unsafe { self.loader.queue_present(queue, &present_info) }
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
