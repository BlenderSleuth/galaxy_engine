// Copyright (c) 2024. Ben Sutherland

use std::mem::ManuallyDrop;
use std::slice;

use app::AppInfo;
use arrayvec::ArrayVec;
use ash::vk;
use nalgebra as na;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};
use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::app;
use crate::engine::MainLoopError::VulkanError;
use crate::maths::ModelViewProjection;
use crate::mesh::{Mesh, MeshError};
use crate::particles::GpuParticleSystem;
use crate::static_resources::{StaticResources, StaticResourcesGuard, StaticResourcesLock};
use crate::uniform_buffer::VolatileUniformBuffer;
use crate::vulkan::command_buffer::{PersistentCmdBufError, ResettablePrimaryCommandPool, TransientPrimaryCommandPool};
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync::{BinarySemaphore, Fence, Semaphore, WaitSemaphore};
use crate::vulkan::{device, instance};

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum EngineInitError {
    #[error("Engine is already initialised.")]
    AlreadyInitialised,
    #[error("Instance init error: {0}")]
    InstanceError(#[from] instance::InstanceInitError),
    #[error("Device init error: {0}")]
    DeviceInitError(#[from] device::DeviceInitError),
    #[error("Vulkan call failed: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Mesh error: {0}")]
    MeshError(#[from] MeshError),
}

#[derive(thiserror::Error, Debug)]
pub enum MainLoopError {
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Command buffer state error: {0}")]
    PersistentCommandBufferError(#[from] PersistentCmdBufError),
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct UniformData {
    sun_direction: na::Vector3<f32>,
    delta_time: f32,
}

// Static resources. These are available while the engine instance is alive.
static STATIC_RESOURCES: StaticResourcesLock = StaticResourcesLock::new(None);

/// Requires that the engine is alive. Currently using parking_lot as a stable polyfill for `MappedRwLockReadGuard`.
pub fn static_resources() -> MappedRwLockReadGuard<'static, StaticResources> {
    RwLockReadGuard::map(STATIC_RESOURCES.read(), |r| r.as_ref().unwrap())
}

pub struct GalaxyEngine {
    surface: ManuallyDrop<Surface>,
    swapchain: ManuallyDrop<Swapchain>,
    static_resources_guard: ManuallyDrop<StaticResourcesGuard>,
    mesh: ManuallyDrop<Mesh>,
    descriptor_pool: ManuallyDrop<DescriptorPool>,
    graphics_cmd_pools: ArrayVec<ResettablePrimaryCommandPool, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_cmd_pools: ArrayVec<ResettablePrimaryCommandPool, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    transient_cmd_pool: ManuallyDrop<TransientPrimaryCommandPool>,
    uniform_buffer: ManuallyDrop<VolatileUniformBuffer>,
    particle_system: ManuallyDrop<GpuParticleSystem>,
    image_available_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    render_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    in_flight_fences: ArrayVec<Fence, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_in_flight_fences: ArrayVec<Fence, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    current_frame: u32,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    window_size: vk::Extent2D,
    window_resized: bool,
    device: ManuallyDrop<Device>,
    instance: Instance,
}

impl GalaxyEngine {
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    pub(crate) const MAX_NUM_PARTICLES: u32 = 1024;

    // TODO: Compute cleanup:
    // - Model resources/descriptor set management.

    // TODO: General cleanup:
    // - RAII handles.
    // - Queue object.
    // - Command buffer and pool management.
    // - Use a single buffer for both vertices and indices.
    pub fn new(
        app_info: &AppInfo,
        display: DisplayHandle,
        window: WindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, EngineInitError> {
        // Currently the engine can only be initialised once.
        static ONCE: std::sync::Once = std::sync::Once::new();
        if ONCE.is_completed() {
            return Err(EngineInitError::AlreadyInitialised);
        }
        ONCE.call_once(|| {});

        // Create instance.
        let instance = Instance::new(app_info, display)?;

        // Create surface.
        let surface = Surface::new(&instance, display, window)?;

        // Create vulkan device. This sets the static device.
        let device = Device::new(instance.loader(), &surface)?;

        // Create transient command pool.
        let mut transient_cmd_pool = TransientPrimaryCommandPool::new(&device, device.primary_queue())?;

        // Initialise engine static resources.
        *STATIC_RESOURCES.write() = Some(StaticResources::new(&device, &mut transient_cmd_pool)?);
        let static_resources_guard = StaticResourcesGuard::new(&STATIC_RESOURCES);

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, &mut transient_cmd_pool, &surface, window_size, None)?;

        // Create uniform buffer.
        let uniform_buffer = VolatileUniformBuffer::new_for_type::<UniformData>("Uniform buffer", &device)?;

        // Create descriptor pool.
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(6),
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1 + 2); // Allocator 1 set for graphics and 2 sets for compute.

        let mut descriptor_pool = DescriptorPool::new(&device, &descriptor_pool_info)?;

        // Load mesh.
        let mesh = Mesh::new(
            "Viking room",
            &device,
            &mut transient_cmd_pool,
            "galaxy_engine/assets/viking_room.obj",
            "galaxy_engine/assets/viking_room.ktx2",
            swapchain.samples(),
            &uniform_buffer,
            &mut descriptor_pool,
        )?;

        // Create particle system.
        let particle_system = GpuParticleSystem::new(
            &device,
            swapchain.samples(),
            Self::MAX_NUM_PARTICLES,
            window_size,
            &uniform_buffer,
            &mut transient_cmd_pool,
            &mut descriptor_pool,
        )?;

        // Create per-frame objects.
        let mut graphics_cmd_pools = ArrayVec::new();
        let mut compute_cmd_pools = ArrayVec::new();

        let mut image_available_semaphores = ArrayVec::new();
        let mut render_finished_semaphores = ArrayVec::new();
        let mut compute_finished_semaphores = ArrayVec::new();
        let mut in_flight_fences = ArrayVec::new();
        let mut compute_in_flight_fences = ArrayVec::new();
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            // Create command pools and buffers.
            let mut graphics_cmd_pool = ResettablePrimaryCommandPool::new(&device, device.primary_queue())?;
            let mut compute_cmd_pool = ResettablePrimaryCommandPool::new(&device, device.primary_queue())?;
            graphics_cmd_pool.allocate_cmd_buffer(vk::CommandBufferLevel::PRIMARY)?;
            compute_cmd_pool.allocate_cmd_buffer(vk::CommandBufferLevel::PRIMARY)?;
            graphics_cmd_pools.push(graphics_cmd_pool);
            compute_cmd_pools.push(compute_cmd_pool);

            // Create sync objects.
            image_available_semaphores.push(BinarySemaphore::new(&device)?);
            render_finished_semaphores.push(BinarySemaphore::new(&device)?);
            compute_finished_semaphores.push(BinarySemaphore::new(&device)?);
            in_flight_fences.push(Fence::new(&device, true)?);
            compute_in_flight_fences.push(Fence::new(&device, true)?);
        }

        device.print_allocator_report();

        Ok(Self {
            instance,
            surface: ManuallyDrop::new(surface),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            static_resources_guard: ManuallyDrop::new(static_resources_guard),
            mesh: ManuallyDrop::new(mesh),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            particle_system: ManuallyDrop::new(particle_system),
            graphics_cmd_pools,
            compute_cmd_pools,
            transient_cmd_pool: ManuallyDrop::new(transient_cmd_pool),
            uniform_buffer: ManuallyDrop::new(uniform_buffer),
            image_available_semaphores,
            render_finished_semaphores,
            compute_finished_semaphores,
            in_flight_fences,
            compute_in_flight_fences,
            current_frame: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            window_size,
            window_resized: false,
        })
    }

    pub fn main_loop(&mut self) -> Result<(), MainLoopError> {
        if self.window_resized {
            self.window_resized = false;
            self.recreate_swapchain()?;
        }

        let loader = self.device.loader();
        let ext = self.device.extensions();

        let current_frame = self.current_frame as usize;

        // Update uniform buffer.
        let time = self.start_time.elapsed().as_secs_f32();
        self.mesh.mvp = ModelViewProjection::spin(self.window_size, time.sin() * 0.5, 20.0);

        let delta_time = self.last_frame_time.elapsed().as_secs_f32();
        self.last_frame_time = std::time::Instant::now();

        let uniform_data = UniformData {
            sun_direction: na::Vector3::new(time.sin().abs(), (time + 0.3).sin().abs(), (time + 0.6).sin().abs()),
            delta_time,
        };

        // Copy data to uniform buffer. TODO: This only works here because the uniform buffer is device-local.
        self.uniform_buffer
            .update(current_frame, bytemuck::bytes_of(&uniform_data))?;

        // Wait for compute fence.
        self.compute_in_flight_fences[current_frame].wait(u64::MAX)?;
        self.compute_in_flight_fences[current_frame].reset()?;

        // Reset compute pool.
        let compute_cmd_pool = &mut self.compute_cmd_pools[current_frame];
        compute_cmd_pool.reset()?;

        let command_buffer = compute_cmd_pool.get_cmd_buffer(0);
        let recording = command_buffer.begin()?;
        self.uniform_buffer
            .copy_to_gpu(loader, current_frame, recording.handle());
        self.particle_system.record_compute(recording);
        command_buffer.end()?;

        let signal_semaphores = [self.compute_finished_semaphores[current_frame].handle()];
        command_buffer.submit(
            &[],
            &signal_semaphores,
            Some(&self.compute_in_flight_fences[current_frame]),
        )?;

        // Wait for graphics fence.
        self.in_flight_fences[current_frame].wait(u64::MAX)?;

        // Acquire image from swapchain.
        let (image_idx, _is_suboptimal) = match self.swapchain.acquire_next_image(
            self.image_available_semaphores[current_frame].handle(),
            vk::Fence::null(),
        ) {
            Ok(x) => x,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => Err(err)?,
        };

        self.in_flight_fences[current_frame].reset()?;

        let swapchain_extent = self.swapchain.get_extent();

        let viewport = vk::Viewport::default()
            .width(swapchain_extent.width as f32)
            .height(swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default().extent(swapchain_extent);

        let color_optimal_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::empty())
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(Swapchain::get_subresource_range());

        // Record graphics command buffer.
        let graphics_cmd_pool = &mut self.graphics_cmd_pools[current_frame];
        graphics_cmd_pool.reset()?;
        let command_buffer = graphics_cmd_pool.get_cmd_buffer(0);

        let recording = command_buffer.begin()?;
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&color_optimal_transition));
        unsafe { ext.sync2.cmd_pipeline_barrier2(recording.handle(), &dependency_info) };

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_colour_resolve_view().view().handle())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            })
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(self.swapchain.get_image_views()[image_idx as usize].handle())
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_depth_view().handle())
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info);

        unsafe { ext.dyn_cmd.cmd_begin_rendering(recording.handle(), &rendering_info) }
        self.particle_system.record_graphics(recording, time, viewport, scissor);
        self.mesh.record_graphics(loader, recording.handle(), viewport, scissor);
        unsafe { ext.dyn_cmd.cmd_end_rendering(recording.handle()) };

        let color_optimal_to_present_src_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(Swapchain::get_subresource_range());

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&color_optimal_to_present_src_transition));
        unsafe { ext.sync2.cmd_pipeline_barrier2(recording.handle(), &dependency_info) };

        command_buffer.end()?;

        // Submit command buffer.
        let wait_semaphores = [
            WaitSemaphore {
                handle: self.compute_finished_semaphores[current_frame].handle(),
                stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
            },
            WaitSemaphore {
                handle: self.image_available_semaphores[current_frame].handle(),
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
        ];
        let signal_semaphores = [self.render_finished_semaphores[current_frame].handle()];
        command_buffer.submit(
            &wait_semaphores,
            &signal_semaphores,
            Some(&self.in_flight_fences[current_frame]),
        )?;

        match self.swapchain.queue_present(
            self.device.primary_queue().handle(),
            image_idx,
            &[self.render_finished_semaphores[current_frame].handle()],
        ) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain()?;
            }
            Err(e) => return Err(VulkanError(e)),
        }

        self.current_frame = (self.current_frame + 1) % Self::MAX_FRAMES_IN_FLIGHT as u32;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> MemResult<()> {
        unsafe { self.device.loader().device_wait_idle() }?;
        let new_swapchain = Swapchain::new(
            &self.instance,
            &self.device,
            &mut self.transient_cmd_pool,
            &self.surface,
            self.window_size,
            Some(&self.swapchain),
        )?;
        let mut old_swapchain = std::mem::replace(&mut self.swapchain, ManuallyDrop::new(new_swapchain));
        unsafe { ManuallyDrop::drop(&mut old_swapchain) };
        Ok(())
    }

    pub fn notify_window_resize(&mut self, width: u32, height: u32) {
        let window_size = vk::Extent2D { width, height };
        if self.window_size == window_size {
            return;
        }
        self.window_size = window_size;
        self.window_resized = true;
    }
}

impl Drop for GalaxyEngine {
    fn drop(&mut self) {
        self.device.print_allocator_report();

        self.device
            .wait_idle()
            .unwrap_or_else(|e| log::error!("Failed to wait for device idle: {:?}", e));

        // Drop sync objects.
        self.image_available_semaphores.clear();
        self.render_finished_semaphores.clear();
        self.compute_finished_semaphores.clear();
        self.in_flight_fences.clear();
        self.compute_in_flight_fences.clear();

        // Drop command_pools.
        self.graphics_cmd_pools.clear();
        self.compute_cmd_pools.clear();
        unsafe { ManuallyDrop::drop(&mut self.transient_cmd_pool) };

        // Drop particle system.
        unsafe { ManuallyDrop::drop(&mut self.particle_system) };

        // Drop model.
        unsafe { ManuallyDrop::drop(&mut self.mesh) };

        // Drop descriptor pool.
        unsafe { ManuallyDrop::drop(&mut self.descriptor_pool) };

        // Drop uniform buffers.
        unsafe { ManuallyDrop::drop(&mut self.uniform_buffer) };

        // Drop swapchain.
        unsafe { ManuallyDrop::drop(&mut self.swapchain) };

        // Drop static resources.
        unsafe { ManuallyDrop::drop(&mut self.static_resources_guard) };

        // Drop vulkan.
        unsafe { ManuallyDrop::drop(&mut self.device) };

        // Drop surface.
        unsafe { ManuallyDrop::drop(&mut self.surface) };

        // Instance is automatically dropped.
    }
}
