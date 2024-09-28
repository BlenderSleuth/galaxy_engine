use ash::{khr, vk};
use ash::prelude::VkResult;

use crate::device::Device;

pub struct Swapchain {
    functor: khr::swapchain::Device,
    pub(crate) handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(instance: &ash::Instance, device: &Device, surface: vk::SurfaceKHR, old_swapchain: Option<&Swapchain>) -> VkResult<Self> {
        let device_properties = device.get_properties();
        let unique_queue_families = device_properties.get_unique_queue_families();

        // Create swapchain.
        let (image_sharing_mode, queue_family_indices) = if unique_queue_families.len() > 1 {
            (vk::SharingMode::CONCURRENT, unique_queue_families)
        } else {
            (vk::SharingMode::EXCLUSIVE, Vec::new())
        };

        let functor = khr::swapchain::Device::new(&instance, &device.device());
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(device_properties.image_count)
            .image_format(device_properties.swapchain_format.format)
            .image_color_space(device_properties.swapchain_format.color_space)
            .image_extent(device_properties.swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(device_properties.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(device_properties.presentation_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.map_or(vk::SwapchainKHR::null(), |swapchain| swapchain.handle));

        let swapchain = unsafe { functor.create_swapchain(&swapchain_info, None) }?;

        // Get swapchain images.
        let images = unsafe { functor.get_swapchain_images(swapchain) }?;

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
            unsafe { device.device().create_image_view(&image_view_info, None) }
        }).collect::<VkResult<Vec<_>>>()?;

        Ok(Self { functor, handle: swapchain, images, image_views })
    }
    
    pub fn get_functor(&self) -> &khr::swapchain::Device {
        &self.functor
    }
    
    pub fn get_image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }
    
    pub unsafe fn destroy(&mut self, device: &Device) {
        // Drop image views.
        for image_view in self.image_views.iter() {
            unsafe { device.device().destroy_image_view(*image_view, None) };
        }
        self.image_views.clear();
        self.images.clear();

        // Drop swapchain (also drops images).
        unsafe { self.functor.destroy_swapchain(self.handle, None) };
    }
}