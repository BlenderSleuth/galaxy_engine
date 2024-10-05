use std::slice;
use ash::{khr, vk};
use ash::prelude::VkResult;
use crate::command_buffer::CommandBuffer;
use crate::device::{Device, PhysicalDeviceProperties, PropertyQueueList};
use crate::engine::MemResult;
use crate::image::{Image, ImageView, ImageWithView};
use crate::surface::Surface;
use crate::utils;

pub struct Swapchain {
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<ImageView<'static>>,
    depth_image_view: Option<ImageWithView>,
    extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(instance: &ash::Instance, device: &Device, gfx_cmd_pool: vk::CommandPool, surface: &Surface, window_size: vk::Extent2D, old_swapchain: Option<&Swapchain>) -> MemResult<Self> {
        let device_properties = device.get_properties();
        let unique_queue_families = device_properties.get_unique_queue_families();

        // Create swapchain.
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

        let loader = khr::swapchain::Device::new(&instance, &device.device());
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(device_properties.image_count)
            .image_format(device_properties.swapchain_format.format)
            .image_color_space(device_properties.swapchain_format.color_space)
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
            .format(device_properties.swapchain_format.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_views = images.iter().map(|swapchain_image| {
            image_view_info.image = *swapchain_image;
            let image_view = unsafe { device.device().create_image_view(&image_view_info, None) }?;
            ImageView::new_external(*swapchain_image, image_view)
        }).collect::<VkResult<Vec<_>>>()?;

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
            .samples(vk::SampleCountFlags::TYPE_1);
        let depth_subresource = vk::ImageSubresourceRange {
            aspect_mask: utils::get_aspect_for_format(PhysicalDeviceProperties::DEPTH_STENCIL_FORMAT),
            ..utils::DEFAULT_SUBRESOURCE_RANGE
        };

        let mut depth_image = Image::new(&device, &depth_image_info, depth_subresource)?;
        depth_image.transition_layout(
            &device,
            CommandBuffer::one_time_transient(&device, gfx_cmd_pool)?,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            None,
            None,
            None,
        )?;

        let depth_image_view = Some(ImageWithView::from_image(&device, depth_image)?);

        Ok(Self { loader, handle, images, image_views, depth_image_view, extent: swapchain_extent })
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn get_images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn get_image_views(&self) -> &[ImageView] {
        &self.image_views
    }

    pub fn get_depth_view(&self) -> &ImageView {
        self.depth_image_view.as_ref().unwrap().borrow_view()
    }

    pub fn get_subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
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

    pub unsafe fn destroy(&mut self, device: &Device) {
        // Drop depth image and view.
        if let Some(depth_image) = self.depth_image_view.take() {
            unsafe { depth_image.destroy(device) };
        }

        // Drop image views.
        for image_view in self.image_views.iter() {
            unsafe { device.device().destroy_image_view(*image_view.handle(), None) };
        }
        self.image_views.clear();
        self.images.clear();

        // Drop swapchain (also drops images).
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}