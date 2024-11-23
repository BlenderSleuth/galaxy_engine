// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::slice;
use std::sync::Arc;

use app::AppInfo;
use arrayvec::ArrayVec;
use ash::vk;
use itertools::izip;
use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard};
use raw_window_handle::{DisplayHandle, WindowHandle};
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{Key, SmolStr};

use crate::camera::{Camera, FirstPersonCamera};
use crate::engine::MainLoopError::VulkanError;
use crate::materials::{Material, MaterialData, MaterialError};
use crate::mesh::{Mesh, MeshError};
use crate::pipelines::PipelineManager;
use crate::prelude::*;
use crate::static_resources::{StaticResources, StaticResourcesGuard, StaticResourcesLock};
use crate::texture::Texture;
use crate::volatile_buffer::{VolatileBuffer, VolatileBufferType};
use crate::vulkan::command_buffer::{
    CmdBufStateTransitionError, ResettablePrimaryCommandPool, TransientPrimaryCommandPool,
};
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::descriptors::{DescriptorPool, DescriptorSetLayout};
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::{MemResult, MemoryError};
use crate::vulkan::image::Sampler;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync::{BinarySemaphore, Semaphore, WaitSemaphore};
use crate::vulkan::{device, instance};
use crate::{app, pipelines};

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SceneUniformData {
    view: Mat4,
    proj: Mat4,
    sun_direction: Vec3,
    delta_time: f32,
}
pub type SceneUniformBuffer = VolatileBuffer<SceneUniformData>;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DrawData {
    pub transform_index: u32,
    pub material_index: u32,
}

pub struct SceneBuffers {
    transforms: VolatileBuffer<[Mat4; 1024]>,
    materials: VolatileBuffer<[MaterialData; 1024]>,
}

impl SceneBuffers {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Mat4, &mut MaterialData)> {
        izip!(self.transforms.local.iter_mut(), self.materials.local.iter_mut())
    }
    pub fn copy_to_gpu(&mut self, frame: usize) -> Result<(), MemoryError> {
        self.transforms.copy_to_gpu(frame)?;
        self.materials.copy_to_gpu(frame)
    }
    pub fn buffer_infos<const N: usize>(&self) -> [[vk::DescriptorBufferInfo; 2]; N] {
        core::array::from_fn(|frame| {
            [
                self.transforms.descriptor_buffer_info(frame),
                self.materials.descriptor_buffer_info(frame),
            ]
        })
    }
}

// Static resources. These are available while the engine instance is alive.
static STATIC_RESOURCES: StaticResourcesLock = StaticResourcesLock::new(None);

/// Requires that the engine is alive. Currently using parking_lot as a stable polyfill for `MappedRwLockReadGuard`.
pub fn static_resources() -> MappedRwLockReadGuard<'static, StaticResources> {
    RwLockReadGuard::map(STATIC_RESOURCES.read(), |r| r.as_ref().unwrap())
}

pub struct GalaxyEngine {
    _static_resources_guard: StaticResourcesGuard,
    meshes: Vec<Mesh>,
    material: Arc<Material>,
    primary_cmd_pools: ArrayVec<ResettablePrimaryCommandPool<2>, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    transient_cmd_pool: TransientPrimaryCommandPool,
    scene_descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    scene_uniform_buffer: SceneUniformBuffer,
    scene_buffers: Box<SceneBuffers>,
    default_sampler: Sampler,
    texture: Arc<Texture>,
    //particle_system: ManuallyDrop<GpuParticleSystem>,
    image_available_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    render_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    compute_finished_semaphores: ArrayVec<BinarySemaphore, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    current_frame: u32,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    window_size: vk::Extent2D,
    window_resized: bool,
    accumulated_mouse_delta: Vec2,
    camera: Camera,
    key_input: HashMap<SmolStr, ElementState>,
    // These are at the bottom so they get dropped last.
    swapchain: Swapchain,
    pipeline_manager: PipelineManager,
    device: Device,
    surface: Surface,
    instance: Instance,
}

impl GalaxyEngine {
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    pub const NUM_MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_4;
    pub const SHADER_PATH: &'static str = "galaxy_engine/content/shaders/";
    pub const NUM_TEXTURES: usize = 2;

    pub(crate) fn new(
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
        let mut transient_cmd_pool =
            TransientPrimaryCommandPool::new("Transient Command Pool", &device, device.primary_queue())?;

        // Initialise engine static resources.
        *STATIC_RESOURCES.write() = Some(StaticResources::new(&device, &mut transient_cmd_pool)?);
        let static_resources_guard = StaticResourcesGuard::new(&STATIC_RESOURCES);

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, &mut transient_cmd_pool, &surface, window_size, None)?;

        let pipeline_manager = PipelineManager::new(&device, swapchain.msaa_samples())?;

        // Create default texture sampler.
        let max_anisotropy = device.physical_device().properties.base.limits.max_sampler_anisotropy;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(max_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(vk::LOD_CLAMP_NONE);
        let default_sampler = Sampler::new(&device, &sampler_info)?;

        // Load texture.
        let texture = Arc::new(Texture::new_from_file(
            "Viking room texture",
            "galaxy_engine/content/models/viking_room/viking_room.ktx2",
            &device,
            &mut transient_cmd_pool,
        )?);

        // Set up scene.

        // Create scene uniform buffer.
        let scene_uniform_buffer = VolatileBuffer::new("Scene uniform buffer", &device, VolatileBufferType::Uniform)?;

        let scene_transforms_buffer = VolatileBuffer::new("Transforms buffer", &device, VolatileBufferType::Storage)?;
        // TODO: don't use volatile buffer for material data buffer.
        let scene_material_buffer = VolatileBuffer::new("Material buffer", &device, VolatileBufferType::Storage)?;
        let scene_buffers = Box::new(SceneBuffers {
            transforms: scene_transforms_buffer,
            materials: scene_material_buffer,
        });

        // Create scene descriptor pool.
        let scene_descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32 * 3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((Self::MAX_FRAMES_IN_FLIGHT * Self::NUM_TEXTURES) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
        ];
        let mut scene_descriptor_pool =
            DescriptorPool::<{ Self::MAX_FRAMES_IN_FLIGHT }>::new(&device, &scene_descriptor_pool_sizes)?;

        scene_descriptor_pool.allocate_descriptor_sets(
            &device,
            &[pipeline_manager.scene_descriptor_set_layout.handle(); Self::MAX_FRAMES_IN_FLIGHT],
        )?;

        let uniform_buffer_info: [_; Self::MAX_FRAMES_IN_FLIGHT] =
            core::array::from_fn(|frame| scene_uniform_buffer.descriptor_buffer_info(frame));
        let buffer_infos = scene_buffers.buffer_infos::<{ Self::MAX_FRAMES_IN_FLIGHT }>();

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.image().view().handle())
            .sampler(default_sampler.handle());
        // 2 of the same image.
        let image_infos = [image_info; Self::NUM_TEXTURES];

        let descriptor_writes: ArrayVec<_, { Self::MAX_FRAMES_IN_FLIGHT * 8 }> = scene_descriptor_pool
            .iter()
            .enumerate()
            .flat_map(|(frame, set)| {
                [
                    // Uniform buffer:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(slice::from_ref(&uniform_buffer_info[frame])),
                    // Transforms buffer:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&buffer_infos[frame][0])),
                    // First texture:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&image_infos[0])),
                    // Second texture:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
                        .dst_array_element(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(slice::from_ref(&image_infos[1])),
                    // Material data:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&buffer_infos[frame][1])),
                ]
            })
            .collect();
        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        // Load material.
        let material = Arc::new(Material::new(
            &device,
            &pipeline_manager,
            "galaxy_engine/content/models/viking_room/viking_room.mat.ron",
        )?);

        // Load mesh.
        let mesh = Mesh::new(
            "Viking room",
            &device,
            &mut transient_cmd_pool,
            "galaxy_engine/content/models/viking_room/viking_room.obj",
            Arc::clone(&material),
        )?;

        // Create particle system.
        //const MAX_NUM_PARTICLES: u32 = 1024;
        //let particle_system = GpuParticleSystem::new(
        //    &device,
        //    swapchain.samples(),
        //    MAX_NUM_PARTICLES,
        //    window_size,
        //    &mut setup_cmd_buffer,
        //)?;

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

        // Set up camera.
        let camera_position = Vec3::new(2., 2., 2.);
        let look_at = Mat4::look_at(camera_position, Vec3::zero(), Vec3::unit_z());
        let camera_transform = Isometry3::new(look_at.extract_translation(), look_at.extract_rotation()).inversed();

        let camera = Camera {
            transform: camera_transform,
            aspect: width as f32 / height as f32,
            fov: 45.,
            near: 0.1,
        };

        Ok(Self {
            instance,
            surface,
            device,
            swapchain,
            pipeline_manager,
            _static_resources_guard: static_resources_guard,
            meshes: vec![mesh],
            material,
            //particle_system: ManuallyDrop::new(particle_system),
            primary_cmd_pools,
            transient_cmd_pool,
            scene_descriptor_pool,
            scene_uniform_buffer,
            scene_buffers,
            default_sampler,
            texture,
            image_available_semaphores,
            render_finished_semaphores,
            compute_finished_semaphores,
            current_frame: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            accumulated_mouse_delta: Vec2::zero(),
            camera,
            window_size,
            key_input: HashMap::new(),
            window_resized: false,
        })
    }

    const MAX_FRAME_TIME: f32 = 1.0 / 60.0;

    pub(crate) fn main_loop(&mut self) -> Result<(), MainLoopError> {
        if self.window_resized {
            self.window_resized = false;
            self.recreate_swapchain()?;
        }

        // Frame time calculations.
        let time = self.start_time.elapsed().as_secs_f32();
        let delta_time = self.last_frame_time.elapsed().as_secs_f32().min(Self::MAX_FRAME_TIME);
        self.last_frame_time = std::time::Instant::now();

        let ext = self.device.extensions();

        let current_frame = self.current_frame as usize;

        // Accumulate mouse input.
        let mouse_delta = self.accumulated_mouse_delta;
        self.accumulated_mouse_delta = Vec2::zero();

        // Update camera rotation.
        {
            const ROTATE_SPEED: f32 = 0.1;
            let first_person_mouse = -Vec2::new(mouse_delta.x, mouse_delta.y) * ROTATE_SPEED;
            self.camera.apply_first_person_mouse(first_person_mouse);
        }
        // Update camera position.
        {
            const MOVE_SPEED: f32 = 3.;

            let mut camera_velocity = Vec3::zero();
            if self.is_key_pressed("w") {
                camera_velocity += self.camera.forward();
            }

            if self.is_key_pressed("s") {
                camera_velocity -= self.camera.forward();
            }

            if self.is_key_pressed("a") {
                camera_velocity -= self.camera.right();
            }

            if self.is_key_pressed("d") {
                camera_velocity += self.camera.right();
            }

            if self.is_key_pressed("e") {
                camera_velocity += Vec3::unit_z();
            }

            if self.is_key_pressed("q") {
                camera_velocity -= Vec3::unit_z();
            }

            if camera_velocity.mag_sq() > 1e-6 {
                camera_velocity.normalize();
            }

            self.camera.transform.translation += camera_velocity * MOVE_SPEED * delta_time;
        }

        // Now that the camera has been updated, calculate the view info.
        let view_info = self.camera.view_info();

        // Update uniform buffer.
        self.scene_uniform_buffer.local = SceneUniformData {
            view: view_info.view,
            proj: view_info.projection,
            sun_direction: Vec3::new(time.sin().abs(), (time + 0.3).sin().abs(), (time + 0.6).sin().abs()),
            delta_time,
        };

        // Update mesh data.
        for (i, (mesh, (transform, material_data))) in self.meshes.iter().zip(self.scene_buffers.iter_mut()).enumerate()
        {
            let i = i as u32;
            *transform = view_info.mvp_from_similarity(&mesh.transform);
            material_data.texture_index = i;
        }

        // Wait for fences of the buffered frame.
        self.primary_cmd_pools[current_frame]
            .get_cmd_buffer(0)
            .wait_for_fence()?;
        self.primary_cmd_pools[current_frame]
            .get_cmd_buffer(1)
            .wait_for_fence()?;
        // Reset command pool.
        let primary_cmd_pool = &mut self.primary_cmd_pools[current_frame];
        primary_cmd_pool.reset()?;

        // Copy uniform buffer to GPU.
        self.scene_uniform_buffer.copy_to_gpu(current_frame)?;
        self.scene_buffers.copy_to_gpu(current_frame)?;

        let compute_cmd_buffer = primary_cmd_pool.get_cmd_buffer(0);
        let _recording = compute_cmd_buffer.begin()?;
        //self.particle_system.record_compute(recording);
        compute_cmd_buffer.end()?;

        let signal_semaphores = [self.compute_finished_semaphores[current_frame].handle()];
        compute_cmd_buffer.submit(&[], &signal_semaphores)?;

        // Begin graphics command buffer recording.

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

        // Record graphics command buffer.
        let gfx_cmd_buffer = primary_cmd_pool.get_cmd_buffer(1);

        // Transition colour attachment to optimal layout (from present).
        let recording = gfx_cmd_buffer.begin()?;
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&color_optimal_transition));
        recording.pipeline_barrier2(ext, &dependency_info);

        let mut color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_colour_resolve_view(image_idx).handle())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            })
            .resolve_mode(vk::ResolveModeFlags::NONE);
        if self.swapchain.msaa_samples() != vk::SampleCountFlags::TYPE_1 {
            color_attachment_info = color_attachment_info
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
        self.material.bind(rendering);
        rendering.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.material.pipeline_layout(),
            0,
            slice::from_ref(&self.scene_descriptor_pool.get(current_frame)),
            &[],
        );
        self.meshes.iter().for_each(|m| m.record_graphics(rendering));
        //self.particle_system.record_graphics(rendering, &view_info, time, viewport, scissor);
        let recording = gfx_cmd_buffer.end_rendering(ext)?;

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
        gfx_cmd_buffer.submit(&wait_semaphores, &signal_semaphores)?;

        match self.swapchain.queue_present(
            self.device.primary_queue_mut(),
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
        let _ = std::mem::replace(&mut self.swapchain, new_swapchain);
        Ok(())
    }

    fn get_key_state(&self, key: &str) -> ElementState {
        self.key_input.get(key).copied().unwrap_or(ElementState::Released)
    }
    fn is_key_pressed(&self, key: &str) -> bool {
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
