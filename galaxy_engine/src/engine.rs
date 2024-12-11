// Copyright (c) 2024 Ben Sutherland.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::Mutex;

use arrayvec::ArrayVec;
use ash::vk;
use const_format::concatcp;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};
use raw_window_handle::{DisplayHandle, WindowHandle};
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{Key, SmolStr};

use crate::app::AppInfo;
use crate::engine::MainLoopError::VulkanError;
use crate::game::Game;
use crate::level::{DeserializableComponentConfig, Level, LoadResult};
use crate::materials::MaterialError;
use crate::meshes::MeshError;
use crate::pipelines;
use crate::pipelines::PipelineManager;
use crate::prelude::*;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::static_resources::{StaticResources, StaticResourcesGuard, StaticResourcesLock};
use crate::vulkan::command_buffer::{
    CmdBufStateTransitionError, ResettablePrimaryCommandPool, TransientPrimaryCommandPool,
};
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync::{BinarySemaphore, Semaphore, WaitSemaphore};
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
    #[error("Pipeline manager init error: {0}")]
    PipelineManagerInitError(#[from] pipelines::PipelineManagerError),
    #[error("Vulkan call failed: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Mesh error: {0}")]
    MeshError(#[from] MeshError),
    #[error("Material error: {0}")]
    MaterialError(#[from] MaterialError),
}

#[derive(thiserror::Error, Debug)]
pub enum MainLoopError {
    #[error("Vulkan function failed with the error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Command buffer state error: {0}")]
    PersistentCommandBufferError(#[from] CmdBufStateTransitionError),
}

// Static resources. These are available while the engine instance is alive.
static STATIC_RESOURCES: StaticResourcesLock = StaticResourcesLock::new(None);

/// Requires that the engine is alive. Currently using parking_lot as a stable polyfill for `MappedRwLockReadGuard`.
pub fn static_resources() -> MappedRwLockReadGuard<'static, StaticResources> {
    RwLockReadGuard::map(STATIC_RESOURCES.read(), |r| r.as_ref().unwrap())
}

pub struct GalaxyEngine {
    game: RefCell<Box<dyn Game>>,
    game_content_dir: PathBuf,
    level: Mutex<Option<Level>>,
    primary_cmd_pools: ArrayVec<ResettablePrimaryCommandPool<2>, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    transient_cmd_pool: Mutex<TransientPrimaryCommandPool>,
    image_available_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    render_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    _compute_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    frame_index: u32,
    start_time: std::time::Instant,
    game_time: std::time::Duration,
    last_frame_time: std::time::Instant,
    window_size: vk::Extent2D,
    window_resized: bool,
    accumulated_mouse_delta: Vec2,
    key_input: HashMap<SmolStr, ElementState>,
    // These are at the bottom so they get dropped last.
    _static_resources_guard: StaticResourcesGuard,
    pub(crate) pipeline_manager: PipelineManager,
    swapchain: Swapchain,
    pub(crate) device: Device,
    surface: Surface,
    instance: Instance,
}

impl GalaxyEngine {
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    pub const NUM_MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_4;

    // Content directories.
    pub const PKG_PATH: &'static str = concatcp!(env!("CARGO_PKG_NAME"), "/");
    pub const BUILD_DIR: &'static str = "build/";
    pub const CONTENT_DIR: &'static str = "content/";
    pub const CONTENT_PATH: &'static str = concatcp!(GalaxyEngine::PKG_PATH, GalaxyEngine::CONTENT_DIR);
    pub const BUILT_PATH: &'static str = concatcp!(GalaxyEngine::PKG_PATH, GalaxyEngine::BUILD_DIR);

    pub(crate) fn new(
        app_info: &AppInfo,
        display: DisplayHandle,
        window: WindowHandle,
        width: u32,
        height: u32,
        game: Box<dyn Game>,
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
        let mut transient_cmd_pool =
            TransientPrimaryCommandPool::new("Transient Command Pool", &device, device.primary_queue())?;

        // Initialise engine static resources.
        *STATIC_RESOURCES.write() = Some(StaticResources::new(&device, &mut transient_cmd_pool)?);
        let static_resources_guard = StaticResourcesGuard::new(&STATIC_RESOURCES);

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, &mut transient_cmd_pool, &surface, window_size, None)?;

        let pipeline_manager = PipelineManager::new(&device, swapchain.msaa_samples())?;

        // Create per-frame objects.
        let mut primary_cmd_pools = ArrayVec::new();

        let mut image_available_semaphores = ArrayVec::new();
        let mut render_finished_semaphores = ArrayVec::new();
        let mut compute_finished_semaphores = ArrayVec::new();
        for frame in 0..Self::MAX_FRAMES_IN_FLIGHT {
            // Create command pools and buffers.
            let mut primary_cmd_pool = ResettablePrimaryCommandPool::new(
                debug_only_name!("Primary Command Pool {frame}"),
                &device,
                device.primary_queue(),
            )?;
            primary_cmd_pool.allocate_cmd_buffers::<2>(vk::CommandBufferLevel::PRIMARY)?;
            primary_cmd_pools.push(primary_cmd_pool);

            // Create sync objects.
            image_available_semaphores.push(BinarySemaphore::new(&device)?);
            render_finished_semaphores.push(BinarySemaphore::new(&device)?);
            compute_finished_semaphores.push(BinarySemaphore::new(&device)?);
        }

        Ok(Self {
            game: RefCell::new(game),
            game_content_dir: app_info.game_dir.clone(),
            level: Mutex::new(None),
            instance,
            surface,
            device,
            swapchain,
            pipeline_manager,
            _static_resources_guard: static_resources_guard,
            primary_cmd_pools,
            transient_cmd_pool: Mutex::new(transient_cmd_pool),
            image_available_semaphores,
            render_finished_semaphores,
            _compute_finished_semaphores: compute_finished_semaphores,
            frame_index: 0,
            start_time: std::time::Instant::now(),
            game_time: std::time::Duration::default(),
            last_frame_time: std::time::Instant::now(),
            accumulated_mouse_delta: Vec2::zero(),
            window_size,
            key_input: HashMap::new(),
            window_resized: false,
        })
    }

    pub fn game_dir(&self) -> &Path {
        &self.game_content_dir
    }

    pub fn game_time(&self) -> std::time::Duration {
        self.game_time
    }

    pub(crate) fn startup(&mut self) -> anyhow::Result<()> {
        // Run game startup callback.
        self.game.borrow_mut().startup(self)?;

        Ok(())
    }

    pub fn load_level<T: DeserializableComponentConfig>(&self, level_path: ResourcePath) -> LoadResult<()> {
        log::info!(
            "Loading level: {}",
            level_path
                .full_path::<resource_type::Level>(self)
                .canonicalize()?
                .display()
        );

        {
            // Lock transient command pool.
            let mut transient_cmd_pool = self.transient_cmd_pool.lock().unwrap();
            // Lock level.
            let mut level_lock = self.level.lock().unwrap();
            *level_lock = Some(Level::new::<T>(
                level_path,
                self,
                &mut transient_cmd_pool,
                level_lock.take(),
            )?);
        }

        Ok(())
    }

    const MAX_FRAME_TIME: f32 = 1.0 / 60.0;

    pub(crate) fn main_loop(&mut self) -> Result<(), MainLoopError> {
        if self.window_resized {
            self.window_resized = false;
            self.recreate_swapchain()?;
        }

        // Frame time calculations.
        self.game_time = self.start_time.elapsed();
        let delta_time = self.last_frame_time.elapsed().as_secs_f32().min(Self::MAX_FRAME_TIME);
        self.last_frame_time = std::time::Instant::now();

        let frame_index = self.frame_index as usize;

        // Accumulate mouse input.
        let mouse_delta = self.accumulated_mouse_delta;
        self.accumulated_mouse_delta = Vec2::zero();

        // Run level update.
        {
            let mut level_lock = self.level.lock().unwrap();
            if let Some(level) = level_lock.deref_mut() {
                level.update(self, delta_time, mouse_delta);
            };
        }

        // Run game update.
        self.game.borrow_mut().update(delta_time);

        // Wait for fences of the buffered frame.
        self.primary_cmd_pools[frame_index].get_cmd_buffer(0).wait_for_fence()?;
        self.primary_cmd_pools[frame_index].get_cmd_buffer(1).wait_for_fence()?;
        // Reset command pool.
        let primary_cmd_pool = &mut self.primary_cmd_pools[frame_index];
        primary_cmd_pool.reset()?;

        {
            let mut level_lock = self.level.lock().unwrap();
            if let Some(level) = level_lock.deref_mut() {
                level.gpu_update(delta_time, self.game_time, frame_index);
            };
        }

        //let compute_cmd_buffer = primary_cmd_pool.get_cmd_buffer(0);
        //let _recording = compute_cmd_buffer.begin()?;
        //self.particle_system.record_compute(recording);
        //compute_cmd_buffer.end()?;

        //let signal_semaphores = [self.compute_finished_semaphores[frame_index].handle()];
        //compute_cmd_buffer.submit(&[], &signal_semaphores)?;

        // Begin graphics command buffer recording.

        // Acquire image from swapchain.
        let (image_idx, _is_suboptimal) = match self
            .swapchain
            .acquire_next_image(self.image_available_semaphores[frame_index].handle(), vk::Fence::null())
        {
            Ok(x) => x,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => Err(err)?,
        };

        let swapchain_extent = self.swapchain.get_extent();

        let viewport = vk::Viewport::default()
            .width(swapchain_extent.width as f32)
            .height(swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default().extent(swapchain_extent);

        let color_optimal_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(Swapchain::get_subresource_range());

        let ext = self.device.extensions();

        // Record graphics command buffer.
        let gfx_cmd_buffer = primary_cmd_pool.get_cmd_buffer(1);

        // Transition colour attachment to optimal layout (from present).
        let recording = gfx_cmd_buffer.begin()?;
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&color_optimal_transition));
        recording.pipeline_barrier2(ext, &dependency_info);

        let mut color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            });
        if self.swapchain.msaa_samples() == vk::SampleCountFlags::TYPE_1 {
            // Render directly to swapchain image.
            color_attachment_info = color_attachment_info
                .image_view(self.swapchain.get_image_views()[image_idx as usize].handle())
                .store_op(vk::AttachmentStoreOp::STORE);
        } else {
            // Render to MSAA image and resolve to swapchain image.
            color_attachment_info = color_attachment_info
                .image_view(self.swapchain.get_colour_resolve_view(image_idx).handle())
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(self.swapchain.get_image_views()[image_idx as usize].handle())
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_depth_view().handle())
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                // Depth is cleared to 0.0 due to reverse-Z.
                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info);

        let rendering = gfx_cmd_buffer.begin_rendering(ext, &rendering_info)?;
        rendering.set_viewport(viewport);
        rendering.set_scissor(scissor);
        {
            let pipeline_manager = &self.pipeline_manager;
            let level_lock = self.level.lock().unwrap();
            if let Some(level) = level_lock.deref() {
                level.render(pipeline_manager, rendering, frame_index);
            };
        }
        let recording = gfx_cmd_buffer.end_rendering(ext)?;

        // Transition colour attachment to present layout.
        let color_optimal_to_present_src_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(Swapchain::get_subresource_range());

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&color_optimal_to_present_src_transition));
        recording.pipeline_barrier2(ext, &dependency_info);

        gfx_cmd_buffer.end()?;

        // Submit command buffer.
        let wait_semaphores = [
            //WaitSemaphore {
            //    handle: self.compute_finished_semaphores[frame_index].handle(),
            //    stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
            //},
            WaitSemaphore {
                handle: self.image_available_semaphores[frame_index].handle(),
                stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
        ];
        let signal_semaphores = [self.render_finished_semaphores[frame_index].handle()];
        gfx_cmd_buffer.submit(&wait_semaphores, &signal_semaphores)?;

        match self.swapchain.queue_present(
            self.device.primary_queue_mut(),
            image_idx,
            &[self.render_finished_semaphores[frame_index].handle()],
        ) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain()?;
            }
            Err(e) => return Err(VulkanError(e)),
        }

        self.frame_index = (self.frame_index + 1) % Self::MAX_FRAMES_IN_FLIGHT as u32;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> MemResult<()> {
        unsafe { self.device.loader().device_wait_idle() }?;
        let new_swapchain = Swapchain::new(
            &self.instance,
            &self.device,
            self.transient_cmd_pool.get_mut().unwrap(),
            &self.surface,
            self.window_size,
            Some(&self.swapchain),
        )?;
        let _ = std::mem::replace(&mut self.swapchain, new_swapchain);
        Ok(())
    }

    pub fn get_window_aspect(&self) -> f32 {
        self.window_size.width as f32 / self.window_size.height as f32
    }

    pub fn get_key_state(&self, key: &str) -> ElementState {
        self.key_input.get(key).copied().unwrap_or(ElementState::Released)
    }

    pub fn is_key_pressed(&self, key: &str) -> bool {
        self.get_key_state(key) == ElementState::Pressed
    }

    pub(crate) fn notify_window_resize(&mut self, width: u32, height: u32) {
        let window_size = vk::Extent2D { width, height };
        if self.window_size == window_size {
            return;
        }
        self.window_size = window_size;
        self.window_resized = true;
    }

    pub(crate) fn notify_keyboard_input(&mut self, event: &KeyEvent) {
        match &event.logical_key {
            Key::Character(c) => {
                self.key_input.insert(c.clone(), event.state);
            }
            _ => {}
        }
    }

    pub(crate) fn notify_mouse_button(&mut self, _state: ElementState, _button: MouseButton) {}

    pub(crate) fn notify_mouse_motion(&mut self, x: f32, y: f32) {
        self.accumulated_mouse_delta += Vec2::new(x, y);
    }
}

impl Drop for GalaxyEngine {
    fn drop(&mut self) {
        // Wait for device before cleaning up.
        self.device
            .wait_idle()
            .unwrap_or_else(|e| log::error!("Failed to wait for device idle: {:?}", e));
    }
}
