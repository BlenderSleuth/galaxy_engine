// Copyright (c) 2024-2025 Ben Sutherland.

use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::{Arc, Mutex, OnceLock};

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk::Handle;
use ash::{ext, khr, nv, vk, RawPtr};
use castaway::match_type;
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use gpu_allocator::AllocatorDebugSettings;
use itertools::{Either, Itertools};

use crate::utils;
use crate::utils::ArcFinalOwner;
use crate::vulkan::debug;
use crate::vulkan::extensions::DeviceExtensions;
use crate::vulkan::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::vulkan::physical_device::{
    PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceIncompatibility, PropertyQueueList,
};
use crate::vulkan::queue::{queue_type, Queue};
use crate::vulkan::surface::Surface;

// Initialised by the engine.
static DEVICE_LOADER: OnceLock<ash::Device> = OnceLock::new();

/// # Safety
///
/// Can be called once the vulkan is initialised by the engine. Device will be destroyed when the engine is dropped.
pub unsafe fn get_device_loader() -> &'static ash::Device {
    DEVICE_LOADER.get().unwrap()
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceInitError {
    #[error("No physical devices found")]
    NoPhysicalDevices,
    #[error("No compatible physical devices found: {0:?}")]
    NoCompatiblePhysicalDevices(Vec<PhysicalDeviceIncompatibility>),
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Allocator error: {0}")]
    AllocatorError(#[from] gpu_allocator::AllocationError),
}

pub type SharedDeviceLoader = Arc<ash::Device>;

pub struct Device {
    loader: ArcFinalOwner<ash::Device>,
    extensions: DeviceExtensions,
    allocator: ArcFinalOwner<Mutex<Allocator>>,
    primary_queue: Queue<queue_type::PrimaryQueue>,
    async_transfer_queue: Option<Queue<queue_type::AsyncTransferQueue>>,
    async_compute_queue: Option<Queue<queue_type::AsyncComputeQueue>>,
    physical: PhysicalDevice,
}

impl Device {
    pub fn new(instance: &ash::Instance, surface: &Surface) -> Result<Self, DeviceInitError> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        if physical_devices.is_empty() {
            return Err(DeviceInitError::NoPhysicalDevices);
        }

        let mut required_device_extensions = vec![
            khr::swapchain::NAME,
            khr::swapchain_mutable_format::NAME,
            khr::synchronization2::NAME,
            khr::dynamic_rendering::NAME,
        ];

        let mut optional_device_extensions = vec![];

        // MacOS compatibility.
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            required_device_extensions.push(khr::portability_subset::NAME);
        }

        // When compiling with debug info, we need debug extensions.
        if cfg!(feature = "debug_info") {
            required_device_extensions.push(khr::shader_non_semantic_info::NAME);
            optional_device_extensions.push(nv::shader_sm_builtins::NAME);
        }

        let physical_devices = physical_devices.into_iter().map(|physical_device| {
            PhysicalDevice::new(
                instance,
                surface,
                physical_device,
                &required_device_extensions,
                &optional_device_extensions,
            )
        });

        let (compatible_devices, incompatible_devices): (Vec<_>, Vec<_>) =
            physical_devices.partition_map(|device_result| match device_result {
                Ok(device) => Either::Left(device),
                Err(err) => Either::Right(err),
            });

        if compatible_devices.is_empty() {
            return Err(DeviceInitError::NoCompatiblePhysicalDevices(incompatible_devices));
        }

        // Pick the first discrete GPU, otherwise the first compatible device.
        let physical_device = compatible_devices
            .into_iter()
            .find_or_first(|device| device.is_discrete)
            .unwrap();

        let unique_queue_families = physical_device.get_unique_queue_families();

        // Create logical device.
        let queue_infos: PropertyQueueList<_> = unique_queue_families
            .into_iter()
            .map(|unique_queue_family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(unique_queue_family)
                    .queue_priorities(&[1.0])
            })
            .collect();

        // Enable device features.
        let PhysicalDeviceFeatures {
            features,
            mut features11,
            mut features12,
            #[cfg(feature = "debug_info")]
            mut shader_sm_builtins_features_nv,
        } = physical_device.enabled_features;

        // Enable dynamic rendering.
        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        // Enable synchronization2.
        let mut synchronization2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let device_extensions = utils::cstr_to_ptrs(&physical_device.enabled_extensions);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut synchronization2_features);

        #[cfg(feature = "debug_info")]
        {
            device_info = device_info.push_next(&mut shader_sm_builtins_features_nv);
        }

        let device = unsafe { instance.create_device(physical_device.handle, &device_info, None) }?;

        // Ensure that this is the only vulkan device, then set static.
        assert!(DEVICE_LOADER.get().is_none());
        DEVICE_LOADER.get_or_init(|| device.clone());

        // Get queues.
        let primary_queue = Queue::get(&device, physical_device.primary_queue_family_idx);
        let async_transfer_queue = physical_device
            .async_transfer_queue_family_idx
            .map(|queue_family_idx| Queue::get(&device, queue_family_idx));
        let async_compute_queue = physical_device
            .async_compute_queue_family_idx
            .map(|queue_family_idx| Queue::get(&device, queue_family_idx));

        // Set up GPU memory allocator.
        let allocator_debug_settings = if cfg!(feature = "debug_info") {
            AllocatorDebugSettings {
                log_memory_information: false,
                log_leaks_on_shutdown: true,
                store_stack_traces: false,
                log_allocations: false,
                log_frees: false,
                log_stack_traces: false,
            }
        } else {
            AllocatorDebugSettings::default()
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.handle,
            debug_settings: allocator_debug_settings,
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        // Load extensions.
        let extensions = DeviceExtensions::new(instance, &device, &[ext::debug_utils::NAME]);

        // Debug name queues.
        debug::set_object_name_with_ext(&extensions, primary_queue.handle(), "Primary Queue")?;
        if let Some(async_transfer_queue) = &async_transfer_queue {
            debug::set_object_name_with_ext(&extensions, async_transfer_queue.handle(), "Async Transfer Queue")?;
        }
        if let Some(async_compute_queue) = &async_compute_queue {
            debug::set_object_name_with_ext(&extensions, async_compute_queue.handle(), "Async Compute Queue")?;
        }

        Ok(Self {
            loader: ArcFinalOwner::new(device),
            extensions,
            allocator: ArcFinalOwner::new(Mutex::new(allocator)),
            primary_queue,
            async_transfer_queue,
            async_compute_queue,
            physical: physical_device,
        })
    }

    pub fn loader(&self) -> &ash::Device {
        &self.loader
    }
    pub fn cloned_loader(&self) -> SharedDeviceLoader {
        Arc::clone(&self.loader)
    }

    pub fn cloned_allocator(&self) -> SharedAllocator {
        Arc::clone(&self.allocator)
    }

    pub fn extensions(&self) -> &DeviceExtensions {
        &self.extensions
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical
    }

    pub fn primary_queue(&self) -> &Queue<queue_type::PrimaryQueue> {
        &self.primary_queue
    }

    pub fn primary_queue_mut(&mut self) -> &mut Queue<queue_type::PrimaryQueue> {
        &mut self.primary_queue
    }

    pub fn async_transfer_queue(&mut self) -> Option<&mut Queue<queue_type::AsyncTransferQueue>> {
        self.async_transfer_queue.as_mut()
    }

    pub fn async_compute_queue(&mut self) -> Option<&mut Queue<queue_type::AsyncComputeQueue>> {
        self.async_compute_queue.as_mut()
    }

    pub fn allocate_and_bind_memory<H: Handle + 'static>(
        &self,
        desc: &AllocationCreateDesc,
        handle: H,
    ) -> MemResult<ManuallyFreeAllocation> {
        let allocation = ManuallyDrop::new(
            self.allocator
                .lock()
                .map_err(|_| gpu_allocator::AllocationError::Internal("Mutex Poisoned".to_string()))?
                .allocate(desc)?,
        );

        match_type!(handle, {
            vk::Buffer as handle => {
                unsafe {
                    self.loader.bind_buffer_memory(
                        handle,
                        allocation.memory(),
                        allocation.offset(),
                    )
                }?;
            },
            vk::Image as handle => {
                unsafe {
                    self.loader.bind_image_memory(
                        handle,
                        allocation.memory(),
                        allocation.offset(),
                    )
                }?;
            },
            _ => {
                // Can this be a compile time error?
                panic!("Invalid handle type.");
            },
        });

        Ok(allocation)
    }

    pub fn wait_idle(&self) -> VkResult<()> {
        unsafe { self.loader.device_wait_idle() }
    }

    pub fn print_allocator_report(&self) {
        #[cfg(feature = "debug_info")]
        log::info!("{:?}", self.allocator.lock().unwrap().generate_report());
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Drop allocator. Allocator has drop semantics, so we don't need a custom destroy closure.
        unsafe { self.allocator.destroy_as_final(|_| {}) }
            .unwrap_or_else(|_| log::error!("Allocator not final owner."));

        // Drop vulkan.
        unsafe { self.loader.destroy_as_final(|device| device.destroy_device(None)) }
            .unwrap_or_else(|_| log::error!("Device not final owner."));
    }
}

// Extension trait for compatibility with arrayvec.
pub trait VkResultExt {
    unsafe fn set_array_vec_len_on_success<T, const N: usize>(
        self,
        v: ArrayVec<T, N>,
        len: usize,
    ) -> VkResult<ArrayVec<T, N>>;
}

impl VkResultExt for vk::Result {
    #[inline]
    unsafe fn set_array_vec_len_on_success<T, const N: usize>(
        self,
        mut v: ArrayVec<T, N>,
        len: usize,
    ) -> VkResult<ArrayVec<T, N>> {
        self.result().map(move |()| {
            v.set_len(len);
            v
        })
    }
}

pub trait DeviceExt {
    unsafe fn create_graphics_pipeline(
        &self,
        pipeline_cache: vk::PipelineCache,
        create_info: &vk::GraphicsPipelineCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::Pipeline>;
    unsafe fn allocate_descriptor_sets_av<const N: usize>(
        &self,
        allocate_info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::DescriptorSet, N>>;
    unsafe fn allocate_command_buffer(
        &self,
        cmd_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> VkResult<vk::CommandBuffer>;
    unsafe fn allocate_command_buffers_av<const N: usize>(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::CommandBuffer, N>>;
}

impl DeviceExt for ash::Device {
    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateGraphicsPipelines.html>
    ///
    /// Non-allocating version of create_graphics_pipelines for single pipeline creation.
    #[inline]
    unsafe fn create_graphics_pipeline(
        &self,
        pipeline_cache: vk::PipelineCache,
        create_info: &vk::GraphicsPipelineCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::Pipeline> {
        let mut pipeline = std::mem::MaybeUninit::uninit();
        (self.fp_v1_0().create_graphics_pipelines)(
            self.handle(),
            pipeline_cache,
            1,
            create_info,
            allocation_callbacks.as_raw_ptr(),
            pipeline.as_mut_ptr(),
        )
        .assume_init_on_success(pipeline)
    }

    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateDescriptorSets.html>
    #[inline]
    unsafe fn allocate_descriptor_sets_av<const N: usize>(
        &self,
        allocate_info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::DescriptorSet, N>> {
        assert!(allocate_info.descriptor_set_count <= N as u32);
        let mut desc_set = ArrayVec::new();
        (self.fp_v1_0().allocate_descriptor_sets)(self.handle(), allocate_info, desc_set.as_mut_ptr())
            .set_array_vec_len_on_success(desc_set, allocate_info.descriptor_set_count as usize)
    }

    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateCommandBuffers.html>
    /// Non-allocating version of allocate_command_buffers for single buffer creation.
    #[inline]
    unsafe fn allocate_command_buffer(
        &self,
        cmd_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> VkResult<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(level)
            .command_buffer_count(1);
        let mut buffer = MaybeUninit::uninit();
        (self.fp_v1_0().allocate_command_buffers)(self.handle(), &allocate_info, buffer.as_mut_ptr())
            .assume_init_on_success(buffer)
    }

    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateCommandBuffers.html>
    #[inline]
    unsafe fn allocate_command_buffers_av<const N: usize>(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::CommandBuffer, N>> {
        assert!(allocate_info.command_buffer_count <= N as u32);
        let mut buffers = ArrayVec::new();
        (self.fp_v1_0().allocate_command_buffers)(self.handle(), allocate_info, buffers.as_mut_ptr())
            .set_array_vec_len_on_success(buffers, allocate_info.command_buffer_count as usize)
    }
}
