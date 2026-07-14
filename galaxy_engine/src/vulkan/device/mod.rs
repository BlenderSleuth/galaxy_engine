// Copyright (c) 2024-2025 Ben Sutherland.

use std::cell::Cell;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::{Arc, Mutex, OnceLock};

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk::Handle;
use ash::{RawPtr, ext, khr, nv, vk};
use castaway::match_type;
use gpu_allocator::AllocatorDebugSettings;
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use itertools::{Either, Itertools};

use crate::utils;
use crate::utils::ArcFinalOwner;
use crate::vulkan::debug;
use crate::vulkan::extensions::DeviceExtensions;
use crate::vulkan::gpu_alloc::{ManuallyFreeAllocation, MemResult, SharedAllocator};
use crate::vulkan::physical_device::{
    PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceIncompatibility, QueueArray,
};
use crate::vulkan::surface::Surface;

pub mod physical_device;
pub mod queue;
use queue::{Queue, QueueType, queue_type};

// Initialised by the engine.
static DEVICE_LOADER: OnceLock<ash::Device> = OnceLock::new();

/// # Safety
///
/// Can be called once the vulkan is initialised by the engine. Device will be destroyed when the engine is dropped.
pub unsafe fn get_device_loader() -> &'static ash::Device {
    DEVICE_LOADER.get().unwrap()
}

// Threaded queue handling.
//
// For a queue to be owned by a particular thread, and only be accessible to that thread, it is
// "claimed" by that thread through Device::claim_queue() at startup.
//
// This registers that the queue type has been claimed (in CLAIMED_QUEUE_TYPES),
// and sets the current thread's queue type thread-locally (in THREAD_QUEUE_TYPE).
//
// This setup means that the queue can only be accessed by the thread that claimed it, without mutexes once claimed.
// There is a 1:1 relationship between queue types and threads. A thread can only claim one queue type.
static CLAIMED_QUEUE_TYPES: Mutex<QueueArray<std::any::TypeId>> = Mutex::new(QueueArray::new_const());
thread_local! {
    static THREAD_QUEUE_TYPE: Cell<Option<std::any::TypeId>> = const { Cell::new(None) };
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ThreadQueueError {
    #[error("Queue already borrowed: {0}")]
    AlreadyBorrowed(#[from] std::cell::BorrowMutError),
    #[error("Queue not set")]
    NotSet,
    #[error("Queue type mismatch")]
    TypeMismatch,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ThreadQueueClaimError {
    #[error("Queue already claimed")]
    AlreadyClaimed,
}

// Private queue traits for type-state queue access. This private super-trait of QueueType allows this bit of tomfoolery.
trait GetQueue {
    fn get_queue(device: &Device) -> &Queue<Self>
    where
        Self: QueueType;
}

impl GetQueue for queue_type::PrimaryQueue {
    fn get_queue(device: &Device) -> &Queue<Self> {
        &device.primary_queue
    }
}

impl GetQueue for queue_type::AsyncTransferQueue {
    fn get_queue(device: &Device) -> &Queue<Self> {
        &device.async_transfer_queue
    }
}

impl GetQueue for queue_type::AsyncComputeQueue {
    fn get_queue(device: &Device) -> &Queue<Self> {
        &device.async_compute_queue
    }
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
    pub loader: ArcFinalOwner<ash::Device>,
    pub extensions: DeviceExtensions,
    pub allocator: ArcFinalOwner<Mutex<Allocator>>,
    primary_queue: Queue<queue_type::PrimaryQueue>,
    async_transfer_queue: Queue<queue_type::AsyncTransferQueue>,
    async_compute_queue: Queue<queue_type::AsyncComputeQueue>,
    pub physical: PhysicalDevice,
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

        // Create logical device.
        let queue_infos = physical_device.queue_infos();

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

        let loader = ArcFinalOwner::new(unsafe { instance.create_device(physical_device.handle, &device_info, None) }?);

        // Ensure that this is the only vulkan device, then set static.
        assert!(DEVICE_LOADER.get().is_none());
        DEVICE_LOADER.get_or_init(|| loader.as_ref().clone());

        // Get queues.
        let primary_queue = unsafe { Queue::get(Arc::clone(&loader), &physical_device.primary_queue) };
        let async_transfer_queue = unsafe { Queue::get(Arc::clone(&loader), &physical_device.async_transfer_queue) };
        let async_compute_queue = unsafe { Queue::get(Arc::clone(&loader), &physical_device.async_compute_queue) };

        // Set up GPU memory allocator.
        let mut allocator_debug_settings = AllocatorDebugSettings::default();
        if cfg!(feature = "debug_info") {
            allocator_debug_settings.log_leaks_on_shutdown = true;
        } else {
            allocator_debug_settings.log_leaks_on_shutdown = false;
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: loader.as_ref().clone(), // Allocator copies all the device function pointers (https://github.com/Traverse-Research/gpu-allocator/issues/159).
            physical_device: physical_device.handle,
            debug_settings: allocator_debug_settings,
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        // Load extensions.
        let extensions = DeviceExtensions::new(instance, &loader, &[ext::debug_utils::NAME]);

        // Debug name queues.
        debug::set_object_name_with_ext(&extensions, primary_queue.vk_handle(), "Primary Queue")?;
        debug::set_object_name_with_ext(&extensions, async_transfer_queue.vk_handle(), "Async Transfer Queue")?;
        debug::set_object_name_with_ext(&extensions, async_compute_queue.vk_handle(), "Async Compute Queue")?;

        Ok(Self {
            loader,
            extensions,
            allocator: ArcFinalOwner::new(Mutex::new(allocator)),
            primary_queue,
            async_transfer_queue,
            async_compute_queue,
            physical: physical_device,
        })
    }

    pub fn cloned_loader(&self) -> SharedDeviceLoader {
        Arc::clone(&self.loader)
    }

    pub fn cloned_allocator(&self) -> SharedAllocator {
        Arc::clone(&self.allocator)
    }

    pub(crate) fn claim_queue<Q: QueueType>(&self) -> Result<&Queue<Q>, ThreadQueueClaimError> {
        {
            let mut claimed_types = CLAIMED_QUEUE_TYPES.lock().unwrap();
            if claimed_types.contains(&std::any::TypeId::of::<Q>()) {
                return Err(ThreadQueueClaimError::AlreadyClaimed);
            }
            claimed_types.push(std::any::TypeId::of::<Q>());
        }

        THREAD_QUEUE_TYPE.set(Some(std::any::TypeId::of::<Q>()));

        Ok(Q::get_queue(self))
    }

    fn get_thread_queue<Q: QueueType>(&self) -> Result<&Queue<Q>, ThreadQueueError> {
        let type_id = THREAD_QUEUE_TYPE.get().ok_or(ThreadQueueError::NotSet)?;

        // Runtime checking of queue type.
        if type_id == std::any::TypeId::of::<Q>() {
            Ok(Q::get_queue(self))
        } else {
            Err(ThreadQueueError::TypeMismatch)
        }
    }

    pub fn get_queue<Q: QueueType>(&self) -> &Queue<Q> {
        self.get_thread_queue().unwrap() // Panic on error here, it should be loud.
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
        // Release device from queue objects.
        unsafe { self.primary_queue.release_device() };
        unsafe { self.async_compute_queue.release_device() };
        unsafe { self.async_transfer_queue.release_device() };

        // Drop allocator. Allocator has drop semantics, so we don't need a custom destroy closure.
        unsafe { self.allocator.destroy_as_final(|_| {}) }
            .unwrap_or_else(|_| log::error!("Allocator not final owner."));

        // Drop vulkan.
        unsafe { self.loader.force_destroy_as_final(|device| device.destroy_device(None)) }
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
            unsafe { v.set_len(len) };
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
        unsafe {
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
    }

    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateDescriptorSets.html>
    #[inline]
    unsafe fn allocate_descriptor_sets_av<const N: usize>(
        &self,
        allocate_info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::DescriptorSet, N>> {
        assert!(allocate_info.descriptor_set_count <= N as u32);
        let mut desc_set = ArrayVec::new();
        unsafe {
            (self.fp_v1_0().allocate_descriptor_sets)(self.handle(), allocate_info, desc_set.as_mut_ptr())
                .set_array_vec_len_on_success(desc_set, allocate_info.descriptor_set_count as usize)
        }
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
        unsafe {
            (self.fp_v1_0().allocate_command_buffers)(self.handle(), &allocate_info, buffer.as_mut_ptr())
                .assume_init_on_success(buffer)
        }
    }

    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateCommandBuffers.html>
    #[inline]
    unsafe fn allocate_command_buffers_av<const N: usize>(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> VkResult<ArrayVec<vk::CommandBuffer, N>> {
        assert!(allocate_info.command_buffer_count <= N as u32);
        let mut buffers = ArrayVec::new();
        unsafe {
            (self.fp_v1_0().allocate_command_buffers)(self.handle(), allocate_info, buffers.as_mut_ptr())
                .set_array_vec_len_on_success(buffers, allocate_info.command_buffer_count as usize)
        }
    }
}
