use std::ffi::{c_char, CStr};
use std::fs::File;
use std::io::BufReader;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::slice;
use std::sync::Arc;
use ash::prelude::VkResult;
use ash::{ext, vk};
use meshopt::VertexDataAdapter;
use nalgebra as na;
use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{app, buffer, engine, device, maths, surface, swapchain, utils};
use buffer::Buffer;
use device::QueueFamily;
use engine::MainLoopError::VulkanError;
use app::AppInfo;
use device::Device;
use maths::VkPerspective;
use surface::Surface;
use swapchain::Swapchain;
use device::SharedDeviceLoader;
use crate::buffer::{CpuToGpu, GpuOnly};
use crate::command_buffer::CommandBuffer;
use crate::gpu_alloc::{MemResult, MemoryError};
use crate::image::{Image, ImageDimensions};
use crate::pipeline::{ComputePipeline, ComputePipelineParameters, GraphicsPipeline, GraphicsPipelineParameters, GraphicsShaderStageArray, Pipeline, PipelineLayout};
use crate::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum EngineInitError {
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
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
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

struct DebugMessenger {
    messenger: vk::DebugUtilsMessengerEXT,
    loader: ext::debug_utils::Instance,
}

impl DebugMessenger {
    fn new(entry: &ash::Entry, instance: &ash::Instance) -> VkResult<Self> {
        let debug_utils_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
                    vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::debug_callback));

        let loader = ext::debug_utils::Instance::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_utils_ci, None) }?;

        Ok(Self { messenger, loader })
    }

    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        use std::borrow::Cow;

        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => log::Level::Warn,
        };

        if std::thread::panicking() {
            return vk::FALSE;
        }

        let cd = unsafe { *p_callback_data };

        let message_id_name =
            unsafe { cd.message_id_name_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
        let message = unsafe { cd.message_as_c_str() }.map_or(Cow::Borrowed(""), CStr::to_string_lossy);
        let message_id_number = cd.message_id_number;

        let _ = std::panic::catch_unwind(|| {
            log::log!(level, "{message_type:?} [{message_id_name} (0x{message_id_number:x})]\n\t{message}");
        });

        vk::FALSE
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) };
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    position: na::Vector3<f32>,
    color: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}

impl Vertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct Particle {
    position: na::Vector2<f32>,
    velocity: na::Vector2<f32>,
    color: na::Vector4<f32>,
}

impl Particle {
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Particle>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Particle, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(std::mem::offset_of!(Particle, color) as u32),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct UniformData {
    //    sun_direction: na::Vector3<f32>,
    delta_time: f32,
}

impl UniformData {
    pub fn size() -> vk::DeviceSize {
        std::mem::size_of::<Self>() as vk::DeviceSize
    }
}

struct ModelViewProjection {
    model: maths::Mat4,
    view: maths::Mat4,
    proj: maths::Mat4,
}

impl ModelViewProjection {
    fn spin(window_size: vk::Extent2D, time: f32, rpm: f32) -> Self {
        Self {
            model: na::Rotation3::from_axis_angle(&na::UnitVector3::new_normalize(na::Vector3::new(0., 0., 1.)), time * 360_f32.to_radians() * rpm / 60.).to_homogeneous(),
            view: na::Isometry3::look_at_rh(&na::Point3::new(2., 2., 2.), &na::Point3::new(0., 0., 0.), &na::Vector3::new(0., 0., 1.)).to_homogeneous(),
            proj: na::Perspective3::vk_new(window_size.width as f32 / window_size.height as f32, 45_f32.to_radians(), 0.1, 10.0).to_homogeneous(),
        }
    }

    fn mvp(&self) -> maths::Mat4 {
        self.proj * self.view * self.model
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Obj error: {0}")]
    ObjError(#[from] obj::ObjError),
    #[error("Model vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
}

struct Model {
    loader: SharedDeviceLoader,
    // TODO: Use a single buffer for both vertices and indices.
    vertex_buffer: Buffer<GpuOnly>,
    index_buffer: Buffer<GpuOnly>,
    texture_image: Image,
    sampler: vk::Sampler,
    vertex_shader_module: ShaderModule<VertexShaderStage>,
    fragment_shader_module: ShaderModule<FragmentShaderStage>,
}

impl Model {
    pub const MODEL_PATH: &'static str = "assets/viking_room.obj";
    pub const TEXTURE_PATH: &'static str = "assets/viking_room.ktx2";

    pub fn new(device: &Device, gfx_cmd_pool: vk::CommandPool) -> Result<Self, ModelError> {
        // Load texture.
        let image_file = std::fs::read(Self::TEXTURE_PATH)?;
        let image = ktx2::Reader::new(image_file).unwrap();
        let header = image.header();
        let mip_levels = image.levels().collect::<Vec<_>>();
        let extent = vk::Extent2D { width: header.pixel_width, height: header.pixel_height };
        let texture_image = Image::new_from_mip_levels(
            "Model texture",
            device,
            gfx_cmd_pool,
            &mip_levels,
            ImageDimensions::Type2D(extent),
            header.format.map(utils::ktx_to_vulkan_format).unwrap_or(vk::Format::R8G8B8A8_SRGB),
        )?;

        // Create texture sampler.
        let max_anisotropy = device.get_properties().properties.limits.max_sampler_anisotropy;
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
            .max_lod(0.);
        let sampler = unsafe { device.loader().create_sampler(&sampler_info, None) }?;

        // Load shaders.
        let vertex_shader_code = std::fs::read("shaders/shader.vert.spv")?;
        let fragment_shader_code = std::fs::read("shaders/shader.frag.spv")?;

        let vertex_shader_module = ShaderModule::new(&device, &vertex_shader_code)?;
        let fragment_shader_module = ShaderModule::new(&device, &fragment_shader_code)?;

        // Load model. The obj crate already does indexing for us.
        let obj_model: obj::Obj<obj::TexturedVertex, u32> = obj::load_obj(BufReader::new(File::open(Model::MODEL_PATH)?))?;

        let vertices = obj_model
            .vertices
            .iter()
            .map(|v| Vertex {
                position: na::Vector3::new(v.position[0], v.position[1], v.position[2]),
                color: na::Vector3::new(1.0, 1.0, 1.0),
                tex_coord: na::Vector2::new(v.texture[0], 1.0 - v.texture[1]),
            })
            .collect::<Vec<Vertex>>();

        // Optimize model.
        let (vertex_count, vert_remap) = meshopt::generate_vertex_remap(&vertices, Some(&obj_model.indices));
        let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &vert_remap);
        let mut indices = meshopt::remap_index_buffer(Some(&obj_model.indices), vertex_count, &vert_remap);
        meshopt::optimize_vertex_cache_in_place(&mut indices, vertex_count);
        let vertex_data_adapter = VertexDataAdapter::new(bytemuck::must_cast_slice(&vertices), std::mem::size_of::<Vertex>(), std::mem::offset_of!(Vertex, position)).unwrap();
        meshopt::optimize_overdraw_in_place(&mut indices, &vertex_data_adapter, 1.05);
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

        // Vertex buffer.
        let mut vertex_buffer = Buffer::<GpuOnly>::new_for_typed_data(
            "Model vertex buffer",
            &device,
            &vertices,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;
        vertex_buffer.copy_via_staging_buffer(&device, bytemuck::must_cast_slice(vertices.as_slice()), gfx_cmd_pool, QueueFamily::Graphics)?;

        // Index buffer.
        let mut index_buffer = Buffer::<GpuOnly>::new_for_typed_data(
            "Model index buffer",
            &device,
            &indices,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
        )?;
        index_buffer.copy_via_staging_buffer(&device, bytemuck::must_cast_slice(indices.as_slice()), gfx_cmd_pool, QueueFamily::Graphics)?;

        Ok(Self {
            loader: device.cloned_loader(),
            vertex_buffer,
            index_buffer,
            texture_image,
            sampler,
            vertex_shader_module,
            fragment_shader_module,
        })
    }

    pub fn shader_stages(&self) -> GraphicsShaderStageArray {
        utils::arrayvec_from_array([
            self.vertex_shader_module.stage_info(),
            self.fragment_shader_module.stage_info(),
        ])
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Drop sampler.
        unsafe { self.loader.destroy_sampler(self.sampler, None) };
    }
}

pub struct GalaxyEngine {
    _entry: ash::Entry,
    instance: ManuallyDrop<ash::Instance>,
    debug_messenger: Option<DebugMessenger>,
    surface: ManuallyDrop<Surface>,
    device: ManuallyDrop<Device>,
    swapchain: ManuallyDrop<Swapchain>,
    model: ManuallyDrop<Model>,
    storage_buffers: Vec<Buffer<GpuOnly>>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    compute_descriptor_sets: [vk::DescriptorSet; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    pipeline: ManuallyDrop<GraphicsPipeline>,
    compute_pipeline: ManuallyDrop<ComputePipeline>,
    graphics_cmd_pool: vk::CommandPool,
    compute_cmd_pool: vk::CommandPool,
    transfer_cmd_pool: vk::CommandPool,
    uniform_buffers: Vec<Buffer<CpuToGpu>>,
    cmd_buffers: [vk::CommandBuffer; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    compute_cmd_buffers: [vk::CommandBuffer; 2],
    image_available_semaphores: [vk::Semaphore; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    render_finished_semaphores: [vk::Semaphore; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    compute_finished_semaphores: [vk::Semaphore; 2],
    in_flight_fences: [vk::Fence; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    compute_in_flight_fences: [vk::Fence; 2],
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
    const MAX_FRAMES_IN_FLIGHT: usize = 2;
    const NUM_PARTICLES: u32 = 1024;

    // TODO: Compute cleanup:
    // - Convert to HLSL.
    // - Separate particle graphics pipeline.
    // - Model cmd buffer recording.
    // - Model resources/descriptor set management.
    // - GpuParticleSystem object.
    // - Material object (shader stages).
    // - Compute shader object.

    // TODO: General cleanup:
    // -
    // - RAII handles.
    // - Queue object.
    // - Command buffer and pool management.
    // - Use a single buffer for both vertices and indices.
    pub fn new(app_info: &AppInfo, display: DisplayHandle, window: WindowHandle, width: u32, height: u32) -> Result<Self, EngineInitError> {
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
        let debug_messenger = if app_info.flags.contains(app::AppFlags::DEBUG) {
            Some(DebugMessenger::new(&entry, &instance)?)
        } else {
            None
        };

        // Create surface.
        let surface = Surface::new(&entry, &instance, display, window)?;

        // Create device.
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
        let transfer_cmd_pool = device.create_transient_command_pool(QueueFamily::Transfer)?;

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, graphics_cmd_pool, &surface, window_size, None)?;

        // Load model.
        let model = Model::new(&device, graphics_cmd_pool)?;

        // Create uniform buffers.
        let uniform_buffers: [Buffer<CpuToGpu>; Self::MAX_FRAMES_IN_FLIGHT] = core::array::from_fn(|_| {
            Buffer::new(
                "Uniform buffer",
                &device,
                1,
                std::mem::size_of::<ModelViewProjection>(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::SharingMode::EXCLUSIVE,
            ).unwrap_or_else(|err| panic!("Failed to create uniform buffer: {err}"))
        });

        // Create descriptor set layout.
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_bindings = [ubo_layout_binding, sampler_layout_binding];
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout = unsafe { device.loader().create_descriptor_set_layout(&descriptor_set_layout_info, None) }?;

        // Create descriptor pool.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32 * 2),
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(Self::MAX_FRAMES_IN_FLIGHT as u32 * 2); // Allocator 2 sets for graphics and 2 sets for compute.

        let descriptor_pool = unsafe { device.loader().create_descriptor_pool(&descriptor_pool_info, None) }?;

        // Create descriptor sets.
        let layouts = [descriptor_set_layout; Self::MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?;

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i].handle())
                .offset(0)
                .range(UniformData::size());

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(model.texture_image.view().handle())
                .sampler(model.sampler);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info)),
            ];

            unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };
        }

        // Create push constant range.
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<maths::Mat4>() as u32);

        // Create pipeline layout.
        let pipeline_layout = Arc::new(PipelineLayout::new(&device, &descriptor_set_layout, &push_constant_range)?);

        // Create pipeline.
        let vertex_shader_code = std::fs::read("shaders/particles.vert.spv")?;
        let fragment_shader_code = std::fs::read("shaders/particles.frag.spv")?;
        let vertex_shader_module = ShaderModule::<VertexShaderStage>::new(&device, &vertex_shader_code)?;
        let fragment_shader_module = ShaderModule::<FragmentShaderStage>::new(&device, &fragment_shader_code)?;
        let particle_shader_stages = utils::arrayvec_from_array([
            vertex_shader_module.stage_info(),
            fragment_shader_module.stage_info(),
        ]);
        let pipeline_params = GraphicsPipelineParameters {
            layout: pipeline_layout,
            vertex_binding_description: Particle::binding_description(),
            //vertex_binding_description: Vertex::binding_description(),
            vertex_attribute_descriptions: &Particle::attribute_descriptions(),
            //vertex_attribute_descriptions: &Vertex::attribute_descriptions(),
            shader_stages: particle_shader_stages, // TODO: Separate particle graphics pipeline.
            samples: swapchain.samples(),
        };
        let pipeline = GraphicsPipeline::new(&device, pipeline_params)?;

        // Create command buffer.
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(graphics_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        let command_buffers = unsafe { device.loader().allocate_command_buffers(&command_buffer_info) }?;

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        let compute_command_buffers = unsafe { device.loader().allocate_command_buffers(&command_buffer_info) }?;

        // Create sync objects.
        let mut image_available_semaphores = [vk::Semaphore::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut render_finished_semaphores = [vk::Semaphore::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut compute_finished_semaphores = [vk::Semaphore::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut in_flight_fences = [vk::Fence::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut compute_in_flight_fences = [vk::Fence::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores[i] = unsafe { device.loader().create_semaphore(&Default::default(), None) }?;
            render_finished_semaphores[i] = unsafe { device.loader().create_semaphore(&Default::default(), None) }?;
            compute_finished_semaphores[i] = unsafe { device.loader().create_semaphore(&Default::default(), None) }?;
            in_flight_fences[i] = unsafe { device.loader().create_fence(&fence_info, None) }?;
            compute_in_flight_fences[i] = unsafe { device.loader().create_fence(&fence_info, None) }?;
        }

        // Set up particle system compute pipeline.
        let particle_shader_code = std::fs::read("shaders/particles.comp.spv")?;
        let particle_shader_module = ShaderModule::new(&device, &particle_shader_code)?;

        // Initial particle positions.
        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let initial_particles = (0..Self::NUM_PARTICLES).map(|_| {
            let r = 0.25 * fastrand::f32().sqrt();
            let theta = 2.0 * std::f32::consts::PI * fastrand::f32();
            let x = r * theta.cos() * window_aspect_ratio;
            let y = r * theta.sin();
            let position = r * na::Vector2::new(x, y);
            Particle {
                position,
                velocity: position.normalize() * 0.25,
                color: na::Vector4::new(fastrand::f32(), fastrand::f32(), fastrand::f32(), 1.0),
            }
        }).collect::<Vec<_>>();

        // Copy to staging buffer.
        let mut particle_staging_buffer = Buffer::<CpuToGpu>::new(
            "Particle staging buffer",
            &device,
            1,
            std::mem::size_of::<Particle>() * Self::NUM_PARTICLES as usize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
        )?;
        particle_staging_buffer.copy_into_buffer(bytemuck::cast_slice(&initial_particles), 0)?;

        let cmd_buffer = CommandBuffer::begin_one_time(&device, graphics_cmd_pool)?;
        let shader_storage_buffers = (0..Self::MAX_FRAMES_IN_FLIGHT).map(|_| {
            let mut buffer = Buffer::<GpuOnly>::new_for_typed_data(
                "Particle storage buffer",
                &device,
                &initial_particles,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::SharingMode::EXCLUSIVE,
            )?;
            let buffer_size = buffer.size();
            particle_staging_buffer.copy_to_buffer(cmd_buffer.as_persistent(), &device, &mut buffer, buffer_size, QueueFamily::Graphics)?;
            Ok(buffer)
        }).collect::<MemResult<Vec<_>>>()?;
        cmd_buffer.end_submit_and_wait(&device, device.get_queue(QueueFamily::Graphics))?;

        // Create compute descriptor set layout.
        let compute_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let compute_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&compute_layout_bindings);
        let compute_descriptor_set_layout = unsafe { device.loader().create_descriptor_set_layout(&compute_layout_info, None) }?;

        // Allocate compute descriptor sets.
        let layouts = [compute_descriptor_set_layout; Self::MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let compute_descriptor_sets = unsafe { device.loader().allocate_descriptor_sets(&alloc_info) }?;

        // Write descriptor sets.
        let descriptor_buffer_infos = (0..Self::MAX_FRAMES_IN_FLIGHT).map(|i| {
            [
                vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[i].handle())
                    .offset(0)
                    .range(UniformData::size()),
                vk::DescriptorBufferInfo::default()
                    .buffer(shader_storage_buffers[(i + 1) % Self::MAX_FRAMES_IN_FLIGHT].handle())
                    .offset(0)
                    .range((std::mem::size_of::<Particle>() * Self::NUM_PARTICLES as usize) as vk::DeviceSize),
                vk::DescriptorBufferInfo::default()
                    .buffer(shader_storage_buffers[i].handle())
                    .offset(0)
                    .range((std::mem::size_of::<Particle>() * Self::NUM_PARTICLES as usize) as vk::DeviceSize),
            ]
        }).collect::<Vec<_>>();

        let descriptor_writes = compute_descriptor_sets.iter().zip(descriptor_buffer_infos.iter()).flat_map(|(descriptor_set, buffer_infos)| {
            [
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[0])),
                // Last frame's storage buffer.
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[1])),
                // Current frame's storage buffer.
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_infos[2])),
            ]
        }).collect::<Vec<_>>();
        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        let compute_pipeline_layout = Arc::new(PipelineLayout::new(&device, &compute_descriptor_set_layout, &push_constant_range)?);

        let compute_pipeline_params = ComputePipelineParameters {
            layout: compute_pipeline_layout,
            compute_module: particle_shader_module,
        };
        let compute_pipeline = ComputePipeline::new(&device, compute_pipeline_params)?;

        device.print_allocator_report();

        Ok(Self {
            _entry: entry,
            instance: ManuallyDrop::new(instance),
            debug_messenger,
            surface: ManuallyDrop::new(surface),
            device: ManuallyDrop::new(device),
            swapchain: ManuallyDrop::new(swapchain),
            model: ManuallyDrop::new(model),
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets: descriptor_sets.try_into().unwrap(),
            compute_descriptor_set_layout,
            compute_descriptor_sets: compute_descriptor_sets.try_into().unwrap(),
            pipeline: ManuallyDrop::new(pipeline),
            compute_pipeline: ManuallyDrop::new(compute_pipeline),
            storage_buffers: shader_storage_buffers,
            graphics_cmd_pool,
            compute_cmd_pool,
            transfer_cmd_pool,
            uniform_buffers: uniform_buffers.into(),
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

        let loader = self.device.loader().deref();
        let ext = self.device.ext();

        let current_frame = self.current_frame as usize;

        // Update uniform buffer.
        let time = self.start_time.elapsed().as_secs_f32();
        let mvp = ModelViewProjection::spin(self.window_size, time, 20.0).mvp();

        let delta_time = self.last_frame_time.elapsed().as_secs_f32();
        self.last_frame_time = std::time::Instant::now();

        let uniform_data = UniformData {
            //sun_direction: na::Vector3::new(delta_time.sin().abs(), (delta_time + 0.3).sin().abs(), (delta_time + 0.6).sin().abs()),
            delta_time,
        };
        // Copy data to uniform buffer.
        self.uniform_buffers[current_frame].copy_into_buffer(bytemuck::bytes_of(&uniform_data), 0)?;

        // Wait for compute fence.
        unsafe { loader.wait_for_fences(&[self.compute_in_flight_fences[current_frame]], true, u64::MAX) }?;
        unsafe { loader.reset_fences(&[self.compute_in_flight_fences[current_frame]]) }?;

        let command_buffer = self.compute_cmd_buffers[current_frame];
        unsafe { loader.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;
        unsafe { loader.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) }?;
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline.handle()) };
        unsafe { loader.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline.layout().handle(), 0, slice::from_ref(&self.compute_descriptor_sets[current_frame]), &[]) };
        unsafe { loader.cmd_dispatch(command_buffer, Self::NUM_PARTICLES / 256, 1, 1) };
        unsafe { loader.end_command_buffer(command_buffer) }?;

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(slice::from_ref(&self.compute_finished_semaphores[current_frame]));
        unsafe { loader.queue_submit(self.device.get_queue(QueueFamily::Compute), &[submit_info], self.compute_in_flight_fences[current_frame]) }?;

        // Wait for graphics fence.
        unsafe { loader.wait_for_fences(&[self.in_flight_fences[current_frame]], true, u64::MAX) }?;

        // Acquire image from swapchain.
        let (image_idx, _is_suboptimal) = match self.swapchain.acquire_next_image(self.image_available_semaphores[current_frame], vk::Fence::null()) {
            Ok(x) => x,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => Err(err)?,
        };

        unsafe { loader.reset_fences(&[self.in_flight_fences[current_frame]]) }?;

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
        unsafe { loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle()) };
        //unsafe { loader.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&self.model.vertex_buffer.handle()), slice::from_ref(&0)) };
        unsafe { loader.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&self.storage_buffers[current_frame].handle()), slice::from_ref(&0)) };
        //unsafe { loader.cmd_bind_index_buffer(command_buffer, self.model.index_buffer.handle(), 0, vk::IndexType::UINT32) };
        unsafe { loader.cmd_push_constants(command_buffer, self.pipeline.layout().handle(), vk::ShaderStageFlags::VERTEX, 0, bytemuck::cast_slice(&[mvp])) };
        unsafe { loader.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout().handle(), 0, slice::from_ref(&self.descriptor_sets[current_frame]), &[]) };
        unsafe { loader.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { loader.cmd_set_scissor(command_buffer, 0, &[scissor]) };
        //unsafe { loader.cmd_draw_indexed(command_buffer, self.model.index_buffer.len(), 1, 0, 0, 0) };
        unsafe { loader.cmd_draw(command_buffer, Self::NUM_PARTICLES, 1, 0, 0) };
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
        let wait_semaphores = [self.compute_finished_semaphores[current_frame], self.image_available_semaphores[current_frame]];
        let wait_stages = [vk::PipelineStageFlags::VERTEX_INPUT, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&command_buffer))
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(slice::from_ref(&self.render_finished_semaphores[current_frame]));

        unsafe { loader.queue_submit(self.device.get_queue(QueueFamily::Graphics), slice::from_ref(&submit_info), self.in_flight_fences[current_frame]) }?;

        match self.swapchain.queue_present(self.device.get_queue(QueueFamily::Present), image_idx, &[self.render_finished_semaphores[current_frame]]) {
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

    fn get_instance_layers(entry: &ash::Entry, flags: &app::AppFlags) -> VkResult<Vec<*const c_char>> {
        // Query available layers.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;

        let mut required_layers = Vec::new();
        if flags.contains(app::AppFlags::DEBUG) {
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

    fn get_required_instance_extensions(entry: &ash::Entry, flags: &app::AppFlags, display: DisplayHandle) -> Result<Vec<*const c_char>, InstanceExtensionError> {
        // Query available extensions
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }?;

        // Require platform windowing extensions. 
        // The returned extensions are pointers to static strings, so we can safely convert them back to CStr.
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

        if flags.contains(app::AppFlags::DEBUG) {
            // Add debug messenger extension.
            required_extensions.push(ext::debug_utils::NAME);
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
        self.device.print_allocator_report();

        let device_loader = self.device.loader();

        unsafe { device_loader.device_wait_idle() }.unwrap_or_else(|e| log::error!("Failed to wait for device idle: {:?}", e));

        // TODO: Make all these handles RAII types so Engine::new() doesn't trigger lifetime validations warnings on init error.
        // Drop sync objects.
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            unsafe { device_loader.destroy_semaphore(self.image_available_semaphores[i], None) };
            unsafe { device_loader.destroy_semaphore(self.render_finished_semaphores[i], None) };
            unsafe { device_loader.destroy_semaphore(self.compute_finished_semaphores[i], None) };
            unsafe { device_loader.destroy_fence(self.in_flight_fences[i], None) };
            unsafe { device_loader.destroy_fence(self.compute_in_flight_fences[i], None) };
        }

        // Drop command_buffers.
        unsafe { device_loader.free_command_buffers(self.graphics_cmd_pool, &self.cmd_buffers) };

        // Drop command_pools.
        unsafe { device_loader.destroy_command_pool(self.graphics_cmd_pool, None) };
        unsafe { device_loader.destroy_command_pool(self.compute_cmd_pool, None) };
        unsafe { device_loader.destroy_command_pool(self.transfer_cmd_pool, None) };

        // Drop pipelines.
        unsafe { ManuallyDrop::drop(&mut self.pipeline) };
        unsafe { ManuallyDrop::drop(&mut self.compute_pipeline) };

        // Drop descriptor set layouts.
        unsafe { device_loader.destroy_descriptor_set_layout(self.compute_descriptor_set_layout, None) };
        unsafe { device_loader.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        // Drop descriptor pool.
        unsafe { device_loader.destroy_descriptor_pool(self.descriptor_pool, None) };

        // Drop uniform buffers.
        self.uniform_buffers.clear();
        self.storage_buffers.clear();

        // Drop model.
        unsafe { ManuallyDrop::drop(&mut self.model) };

        // Drop swapchain.
        unsafe { ManuallyDrop::drop(&mut self.swapchain) };

        // Drop device.
        unsafe { ManuallyDrop::drop(&mut self.device) };

        // Drop surface.
        unsafe { ManuallyDrop::drop(&mut self.surface) };

        // Drop debug messenger.
        self.debug_messenger = None;

        // Drop instance.
        unsafe { self.instance.destroy_instance(None) };

        // Entry is automatically dropped.
    }
}