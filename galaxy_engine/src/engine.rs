use std::ffi::{c_char, CStr};
use std::mem::ManuallyDrop;
use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::descriptors::DescriptorPool;
use crate::device::DeviceExt;
use crate::gpu_alloc::{MemResult, MemoryError};
use crate::maths::ModelViewProjection;
use crate::mesh::{Mesh, MeshError};
use crate::particles::GpuParticleSystem;
use crate::static_resources::{StaticResources, StaticResourcesGuard, StaticResourcesLock};
use crate::sync::{BinarySemaphore, Fence};
use crate::uniform_buffer::VolatileUniformBuffer;
use crate::{app, device, engine, surface, swapchain, utils};
use app::AppInfo;
use arrayvec::ArrayVec;
use device::Device;
use device::QueueFamily;
use engine::MainLoopError::VulkanError;
use nalgebra as na;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};
use raw_window_handle::{DisplayHandle, WindowHandle};
use surface::Surface;
use swapchain::Swapchain;


#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum EngineInitError {
    #[error("Engine is already initialised.")]
    AlreadyInitialised,
    #[error("Library load failed: {0}")]
    LibraryLoadFailed(#[from] ash::LoadingError),
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error(
        "App requires Vulkan {}.{}.{} (Current: {}.{}.{}). Consider updating your graphics drivers",
        vk::api_version_major(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_minor(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_patch(GalaxyEngine::MIN_VK_VERSION),
        vk::api_version_major(*.0),
        vk::api_version_minor(*.0),
        vk::api_version_patch(*.0)
    )]
    IncompatibleVulkanVersion(u32),
    #[error("Instance extension error: {0}")]
    InstanceExtensionError(#[from] InstanceExtensionError),
    #[error("Device init error: {0}")]
    DeviceInitError(#[from] device::DeviceInitError),
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
}

#[derive(thiserror::Error, Debug)]
pub enum InstanceExtensionError {
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Unable to find Vulkan extension: {0:?}")]
    ExtensionNotFound(&'static CStr),
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
    _entry: ash::Entry,
    instance: ash::Instance,
    #[cfg(feature = "debug_info")]
    debug_messenger: Option<crate::debug::DebugMessenger>,
    surface: ManuallyDrop<Surface>,
    device: ManuallyDrop<Device>,
    swapchain: ManuallyDrop<Swapchain>,
    static_resources_guard: ManuallyDrop<StaticResourcesGuard>,
    mesh: ManuallyDrop<Mesh>,
    descriptor_pool: ManuallyDrop<DescriptorPool>,
    graphics_cmd_pool: vk::CommandPool,
    compute_cmd_pool: vk::CommandPool,
    transfer_cmd_pool: vk::CommandPool,
    uniform_buffer: ManuallyDrop<VolatileUniformBuffer>,
    particle_system: ManuallyDrop<GpuParticleSystem>,
    cmd_buffers: ArrayVec<vk::CommandBuffer, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_cmd_buffers: ArrayVec<vk::CommandBuffer, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
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
}

impl GalaxyEngine {
    const MIN_VK_VERSION: u32 = vk::make_api_version(0, 1, 2, 0);
    const ENGINE_NAME: &'static CStr = c"Galaxy Engine";
    const ENGINE_VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    pub(crate) const MAX_NUM_PARTICLES: u32 = 1024;

    // TODO: Compute cleanup:
    // - Convert to HLSL.
    // - Model resources/descriptor set management.

    // TODO: General cleanup:
    // - RAII handles.
    // - Queue object.
    // - Command buffer and pool management.
    // - Use a single buffer for both vertices and indices.
    pub fn new(app_info: &AppInfo, display: DisplayHandle, window: WindowHandle, width: u32, height: u32) -> Result<Self, EngineInitError> {
        // Currently the engine can only be initialised once.
        static ONCE: std::sync::Once = std::sync::Once::new();
        if ONCE.is_completed() {
            return Err(EngineInitError::AlreadyInitialised);
        }
        ONCE.call_once(|| {});

        // Setup Vulkan.
        let entry = unsafe { ash::Entry::load() }?;

        // Check Vulkan API version.
        let api_version = unsafe { entry.try_enumerate_instance_version() }?.unwrap_or_else(|| vk::API_VERSION_1_0);

        // Require minimum VK version.
        if api_version < Self::MIN_VK_VERSION {
            return Err(EngineInitError::IncompatibleVulkanVersion(api_version));
        }

        // Get instance extensions and layers
        let layers = Self::get_instance_layers(&entry, &app_info.flags)?;
        let instance_extensions = Self::get_required_instance_extensions(&entry, &app_info.flags, display)?;

        let vk_app_info = vk::ApplicationInfo::default()
            .application_name(&app_info.name)
            .application_version(app_info.version)
            .engine_name(Self::ENGINE_NAME)
            .engine_version(utils::parse_version(Self::ENGINE_VERSION_STR))
            .api_version(Self::MIN_VK_VERSION);

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&vk_app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&instance_extensions)
            .flags(create_flags);

        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        // Create debug messenger.
        #[cfg(feature = "debug_info")]
        let debug_messenger = if app_info.flags.contains(app::AppFlags::DEBUG) {
            Some(crate::debug::DebugMessenger::new(&entry, &instance)?)
        } else {
            None
        };

        // Create surface.
        let surface = Surface::new(&entry, &instance, display, window)?;

        // Create device. This sets the static device.
        let device = Device::new(&instance, &surface)?;
        let device_properties = device.get_properties();

        // Create command pools.
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device_properties.graphics_queue_family_idx);
        let graphics_cmd_pool = unsafe { device.loader().create_command_pool(&command_pool_info, None) }?;
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device_properties.compute_queue_family_idx);
        let compute_cmd_pool = unsafe { device.loader().create_command_pool(&command_pool_info, None) }?;

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.get_queue_family_idx(QueueFamily::Graphics));
        let transfer_cmd_pool = unsafe { device.loader().create_command_pool(&command_pool_info, None) }?;

        // Initialise engine static resources.
        *STATIC_RESOURCES.write() = Some(StaticResources::new(&device, graphics_cmd_pool)?);
        let static_resources_guard = StaticResourcesGuard::new(&STATIC_RESOURCES);

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, graphics_cmd_pool, &surface, window_size, None)?;

        // Create uniform buffer.
        let uniform_buffer = VolatileUniformBuffer::new_for_type::<UniformData>(
            "Uniform buffer",
            &device,
        )?;

        // Create descriptor pool.
        let pool_sizes = [
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
            .pool_sizes(&pool_sizes)
            .max_sets(1 + 2); // Allocator 1 set for graphics and 2 sets for compute.

        let descriptor_pool = DescriptorPool::new(&device, &descriptor_pool_info)?;

        // Load mesh.
        let mesh = Mesh::new(
            "Viking room",
            &device,
            graphics_cmd_pool,
            "galaxy_engine/assets/viking_room.obj",
            "galaxy_engine/assets/viking_room.ktx2",
            swapchain.samples(),
            &uniform_buffer,
            &descriptor_pool,
        )?;

        // Create command buffer.
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(graphics_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        let command_buffers = unsafe { device.loader().allocate_command_buffers_av(&command_buffer_info) }?;

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        let compute_command_buffers = unsafe { device.loader().allocate_command_buffers_av(&command_buffer_info) }?;

        // Create sync objects.
        let mut image_available_semaphores = ArrayVec::new();
        let mut render_finished_semaphores = ArrayVec::new();
        let mut compute_finished_semaphores = ArrayVec::new();
        let mut in_flight_fences = ArrayVec::new();
        let mut compute_in_flight_fences = ArrayVec::new();
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(BinarySemaphore::new(&device)?);
            render_finished_semaphores.push(BinarySemaphore::new(&device)?);
            compute_finished_semaphores.push(BinarySemaphore::new(&device)?);
            in_flight_fences.push(Fence::new(&device, true)?);
            compute_in_flight_fences.push(Fence::new(&device, true)?);
        }

        let particle_system = GpuParticleSystem::new(&device, swapchain.samples(), Self::MAX_NUM_PARTICLES, window_size, &uniform_buffer, graphics_cmd_pool, &descriptor_pool)?;

        device.print_allocator_report();

        Ok(Self {
            _entry: entry,
            instance,
            #[cfg(feature = "debug_info")]
            debug_messenger,
            surface: ManuallyDrop::new(surface),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            static_resources_guard: ManuallyDrop::new(static_resources_guard),
            mesh: ManuallyDrop::new(mesh),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            particle_system: ManuallyDrop::new(particle_system),
            graphics_cmd_pool,
            compute_cmd_pool,
            transfer_cmd_pool,
            uniform_buffer: ManuallyDrop::new(uniform_buffer),
            cmd_buffers: command_buffers.try_into().unwrap(),
            compute_cmd_buffers: compute_command_buffers.try_into().unwrap(),
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
        let ext = self.device.ext();

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
        // Copy data to uniform buffer.
        self.uniform_buffer.update(current_frame, bytemuck::bytes_of(&uniform_data))?;

        // Wait for compute fence.
        self.compute_in_flight_fences[current_frame].wait(u64::MAX)?;
        self.compute_in_flight_fences[current_frame].reset()?;

        let command_buffer = self.compute_cmd_buffers[current_frame];
        unsafe { loader.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;
        unsafe { loader.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) }?;
        self.uniform_buffer.copy_to_gpu(loader, current_frame, command_buffer);
        self.particle_system.record_compute(command_buffer);
        unsafe { loader.end_command_buffer(command_buffer) }?;

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(slice::from_ref(self.compute_finished_semaphores[current_frame].ref_handle()));
        unsafe { loader.queue_submit(self.device.get_queue(QueueFamily::Compute), &[submit_info], self.compute_in_flight_fences[current_frame].handle()) }?;

        // Wait for graphics fence.
        self.in_flight_fences[current_frame].wait(u64::MAX)?;

        // Acquire image from swapchain.
        let (image_idx, _is_suboptimal) = match self.swapchain.acquire_next_image(self.image_available_semaphores[current_frame].handle(), vk::Fence::null()) {
            Ok(x) => x,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => Err(err)?,
        };

        self.in_flight_fences[current_frame].reset()?;

        let command_buffer = self.cmd_buffers[current_frame];

        unsafe { loader.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;

        let swapchain_extent = self.swapchain.get_extent();

        let viewport = vk::Viewport::default()
            .width(swapchain_extent.width as f32)
            .height(swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .extent(swapchain_extent);

        // Record command buffer.
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { loader.begin_command_buffer(command_buffer, &begin_info) }?;

        let color_optimal_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::empty())
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(Swapchain::get_subresource_range());

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&color_optimal_transition));
        unsafe { ext.sync2.cmd_pipeline_barrier2(command_buffer, &dependency_info) };

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_colour_resolve_view().view().handle())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(self.swapchain.get_image_views()[image_idx as usize].handle())
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_depth_view().handle())
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: swapchain_extent })
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info);
        unsafe { ext.dyn_cmd.cmd_begin_rendering(command_buffer, &rendering_info) }
        self.particle_system.record_graphics(command_buffer, time, viewport, scissor);
        self.mesh.record_graphics(loader, command_buffer, viewport, scissor);
        unsafe { ext.dyn_cmd.cmd_end_rendering(command_buffer) };

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
        unsafe { ext.sync2.cmd_pipeline_barrier2(command_buffer, &dependency_info) };

        unsafe { loader.end_command_buffer(command_buffer) }?;

        // Submit command buffer.
        let wait_semaphores = [self.compute_finished_semaphores[current_frame].handle(), self.image_available_semaphores[current_frame].handle()];
        let wait_stages = [vk::PipelineStageFlags::VERTEX_INPUT, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&command_buffer))
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(slice::from_ref(self.render_finished_semaphores[current_frame].ref_handle()));

        unsafe { loader.queue_submit(self.device.get_queue(QueueFamily::Graphics), slice::from_ref(&submit_info), self.in_flight_fences[current_frame].handle()) }?;

        match self.swapchain.queue_present(self.device.get_queue(QueueFamily::Present), image_idx, &[self.render_finished_semaphores[current_frame].handle()]) {
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
        let new_swapchain = Swapchain::new(&self.instance, &self.device, self.graphics_cmd_pool, &self.surface, self.window_size, Some(&self.swapchain))?;
        unsafe { ManuallyDrop::drop(&mut self.swapchain) };
        self.swapchain = ManuallyDrop::new(new_swapchain);
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

    fn get_instance_layers(entry: &ash::Entry, _flags: &app::AppFlags) -> VkResult<Vec<*const c_char>> {
        // Query available layers.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;

        let mut required_layers = Vec::new();
        #[cfg(feature = "debug_info")]
        if _flags.contains(app::AppFlags::DEBUG) {
            required_layers.push(c"VK_LAYER_KHRONOS_validation");
        }

        // Check all required layers are available. Not fatal if not found.
        required_layers.retain(|&required_layer| {
            if available_layers.iter().any(|&available_layer| {
                available_layer.layer_name_as_c_str() == Ok(required_layer)
            }) {
                true
            } else {
                log::warn!("Required layer not found: {:?}.", required_layer);
                false
            }
        });

        Ok(utils::cstr_to_ptrs(&required_layers))
    }

    fn get_required_instance_extensions(entry: &ash::Entry, _flags: &app::AppFlags, display: DisplayHandle) -> Result<Vec<*const c_char>, InstanceExtensionError> {
        // Query available extensions
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }?;

        // Require platform windowing extensions. 
        // The returned extensions are pointers to static strings, so we can safely convert them back to CStr.
        #[allow(unused_mut)]
        let mut required_extensions = ash_window::enumerate_required_extensions(display.as_raw())?
            .iter()
            .map(|&ext| unsafe { CStr::from_ptr(ext) })
            .collect::<Vec<_>>();

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME);
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME);
        }

        #[cfg(feature = "debug_info")]
        if _flags.contains(app::AppFlags::DEBUG) {
            // Add debug messenger extension.
            required_extensions.push(ash::ext::debug_utils::NAME);
        }

        // Check all required extensions are available.
        for required_extension in required_extensions.iter() {
            if !available_extensions.iter().any(|&available_extension| {
                available_extension.extension_name_as_c_str() == Ok(required_extension)
            }) {
                return Err(InstanceExtensionError::ExtensionNotFound(required_extension));
            }
        }

        Ok(utils::cstr_to_ptrs(&required_extensions))
    }
}

impl Drop for GalaxyEngine {
    fn drop(&mut self) {
        let device_loader = self.device.loader();

        unsafe { device_loader.device_wait_idle() }.unwrap_or_else(|e| log::error!("Failed to wait for device idle: {:?}", e));

        self.device.print_allocator_report();

        // Drop sync objects.
        self.image_available_semaphores.clear();
        self.render_finished_semaphores.clear();
        self.compute_finished_semaphores.clear();
        self.in_flight_fences.clear();
        self.compute_in_flight_fences.clear();

        // Drop command_buffers.
        unsafe { device_loader.free_command_buffers(self.graphics_cmd_pool, &self.cmd_buffers) };

        // Drop command_pools.
        unsafe { device_loader.destroy_command_pool(self.graphics_cmd_pool, None) };
        unsafe { device_loader.destroy_command_pool(self.compute_cmd_pool, None) };
        unsafe { device_loader.destroy_command_pool(self.transfer_cmd_pool, None) };

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

        // Drop device.
        unsafe { ManuallyDrop::drop(&mut self.device) };

        // Drop surface.
        unsafe { ManuallyDrop::drop(&mut self.surface) };

        // Drop debug messenger.
        #[cfg(feature = "debug_info")]
        {
            self.debug_messenger = None;
        }

        // Drop instance.
        unsafe { self.instance.destroy_instance(None) };

        // Entry is automatically dropped.
    }
}