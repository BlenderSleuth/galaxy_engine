// Copyright (c) 2024-2025 Ben Sutherland.

use std::mem::ManuallyDrop;
use std::slice;

use ash::prelude::VkResult;
use ash::{khr, vk};

use crate::engine::GalaxyEngine;
use crate::loading::LoadingContext;
use crate::utils;
use crate::utils::linear_from_srgb_format;
use crate::vulkan::device::queue::queue_type::PrimaryQueue;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::image::{Image, ImageView};
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;

pub struct SwapchainImage<'a> {
    pub index: u32,
    pub image: vk::Image,
    pub colour_resolve: vk::ImageView,
    pub colour_resolve_linear: vk::ImageView,
    pub srgb: vk::ImageView,
    pub linear: vk::ImageView,
    _marker: std::marker::PhantomData<&'a Swapchain>,
}

pub struct Swapchain {
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<ImageView>,
    linear_image_views: Vec<ImageView>,
    colour_resolve_image: Option<Image>,
    colour_resolve_linear_view: Option<ImageView>,
    depth_image: ManuallyDrop<Image>,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        loading_ctx: &mut LoadingContext<PrimaryQueue>,
        surface: &Surface,
        window_size: vk::Extent2D,
        old_swapchain: Option<&Swapchain>,
    ) -> MemResult<Self> {
        let physical_properties = &device.physical;

        // Create swapchain.

        let surface_capabilities = surface.get_capabilities(physical_properties.handle)?;

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
        let surface_srgb_format = physical_properties.surface_format;
        let surface_linear_format = physical_properties.surface_linear_format;

        let loader = khr::swapchain::Device::new(instance.loader(), &device.loader);

        let view_formats = [surface_srgb_format.format, surface_linear_format];

        let mut view_format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
            .surface(surface.handle())
            .min_image_count(physical_properties.swapchain_image_count)
            .image_format(surface_srgb_format.format)
            .image_color_space(surface_srgb_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(physical_properties.presentation_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.map_or(vk::SwapchainKHR::null(), |swapchain| swapchain.handle))
            .push_next(&mut view_format_list);

        let handle = unsafe { loader.create_swapchain(&swapchain_info, None) }?;

        // Get swapchain images.
        let images = unsafe { loader.get_swapchain_images(handle) }?;

        // Create image views.
        let mut image_view_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .components(vk::ComponentMapping::default())
            .subresource_range(utils::DEFAULT_SUBRESOURCE_RANGE);

        let mut create_image_views = |images: &[vk::Image], format: vk::Format| -> VkResult<Vec<ImageView>> {
            image_view_info.format = format;
            images
                .iter()
                .map(|swapchain_image| {
                    image_view_info.image = *swapchain_image;
                    unsafe { ImageView::new(device, &image_view_info) }
                })
                .collect::<VkResult<_>>()
        };

        // Create srgb image views.
        let image_views = create_image_views(&images, surface_srgb_format.format)?;

        // Create linear image views.
        let linear_image_views = create_image_views(&images, linear_from_srgb_format(surface_srgb_format.format))?;

        let msaa_samples = if physical_properties
            .supported_msaa_samples
            .contains(GalaxyEngine::NUM_MSAA_SAMPLES)
        {
            GalaxyEngine::NUM_MSAA_SAMPLES
        } else {
            vk::SampleCountFlags::TYPE_1
        };

        // Create colour resolve image for multisampling.
        let colour_resolve_image = if msaa_samples > vk::SampleCountFlags::TYPE_1 {
            let colour_resolve_info = vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .image_type(vk::ImageType::TYPE_2D)
                .extent(swapchain_extent.into())
                .mip_levels(1)
                .array_layers(1)
                .format(surface_srgb_format.format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(msaa_samples)
                .push_next(&mut view_format_list);
            Some(Image::new(
                "Colour resolve image",
                &device,
                &colour_resolve_info,
                Self::get_subresource_range(),
            )?)
        } else {
            None
        };

        let colour_resolve_linear_view = if let Some(image) = &colour_resolve_image {
            let image_view_info = vk::ImageViewCreateInfo::default()
                .image(image.handle())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(linear_from_srgb_format(surface_srgb_format.format))
                .components(vk::ComponentMapping::default())
                .subresource_range(Self::get_subresource_range());
            Some(unsafe { ImageView::new(device, &image_view_info) }?)
        } else {
            None
        };

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
            .format(physical_properties.depth_stencil_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);
        let depth_subresource = vk::ImageSubresourceRange {
            aspect_mask: utils::get_aspect_for_format(physical_properties.depth_stencil_format),
            ..Self::get_subresource_range()
        };

        let mut depth_image = Image::new("Depth image", device, &depth_image_info, depth_subresource)?;

        loading_ctx.load(|cmd_buf| {
            let barrier = depth_image.layout_transition_barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                None,
            );
            let dep_info = vk::DependencyInfoKHR::default().image_memory_barriers(slice::from_ref(&barrier));
            cmd_buf.pipeline_barrier2(device, &dep_info);
            Ok(Vec::new())
        })?;

        Ok(Self {
            loader,
            handle,
            images,
            image_views,
            linear_image_views,
            colour_resolve_image,
            colour_resolve_linear_view,
            depth_image: ManuallyDrop::new(depth_image),
            extent: swapchain_extent,
            msaa_samples,
        })
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    //fn get_colour_resolve_view(&self, image_idx: u32) -> vk::ImageView {}

    //pub fn get_images(&self) -> &[vk::Image] {
    //    &self.images
    //}

    //pub fn get_image_view(&self, index: u32) -> vk::ImageView {
    //    self.image_views[index as usize].handle()
    //}
    //
    //pub fn get_linear_image_view(&self, index: u32) -> vk::ImageView {
    //    self.linear_image_views[index as usize].handle()
    //}

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

    pub fn msaa_samples(&self) -> vk::SampleCountFlags {
        self.msaa_samples
    }

    pub fn acquire_next_image(&self, semaphore: vk::Semaphore, fence: vk::Fence) -> VkResult<(SwapchainImage, bool)> {
        let (index, optimal) = unsafe { self.loader.acquire_next_image(self.handle, u64::MAX, semaphore, fence) }?;
        let i = index as usize;
        Ok((
            SwapchainImage {
                index,
                image: self.images[i],
                colour_resolve: self
                    .colour_resolve_image
                    .as_ref()
                    .map(|image| image.view())
                    .unwrap_or(&self.image_views[i])
                    .handle(),
                colour_resolve_linear: self
                    .colour_resolve_linear_view
                    .as_ref()
                    .map(|view| view.handle())
                    .unwrap_or(self.linear_image_views[i].handle()),
                srgb: self.image_views[i].handle(),
                linear: self.linear_image_views[i].handle(),
                _marker: std::marker::PhantomData,
            },
            optimal,
        ))
    }

    pub(super) unsafe fn queue_present(
        &self,
        queue: vk::Queue,
        image: SwapchainImage,
        wait_semaphores: &[vk::Semaphore],
    ) -> VkResult<bool> {
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(slice::from_ref(&self.handle))
            .image_indices(slice::from_ref(&image.index));

        unsafe { self.loader.queue_present(queue, &present_info) }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // Drop depth image and view.
        unsafe { ManuallyDrop::drop(&mut self.depth_image) };

        // Drop colour resolve image.
        self.colour_resolve_image.take();

        // Drop image views.
        self.image_views.clear();
        self.images.clear();

        // Drop swapchain (also drops images).
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}
