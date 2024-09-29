use ash::prelude::VkResult;
use ash::{ext, khr, vk};
use raw_window_handle::{DisplayHandle, WindowHandle};
use std::ffi::{c_char, CStr};
use std::slice;

use crate::buffer::Buffer;
use crate::device::QueueFamily;
use crate::{app, buffer, device, surface, swapchain, utils};
use app::AppInfo;
use device::Device;
use surface::Surface;
use swapchain::Swapchain;

// Const versions of nalgebra-glm functions. TODO: pull request to nalgebra-glm.
mod glm {
    pub use nalgebra_glm::{Mat4, Vec2, Vec3};
    use nalgebra_glm::{Scalar, TVec2, TVec3, TVec4};

    pub const fn vec2<T: Scalar>(x: T, y: T) -> TVec2<T> {
        TVec2::new(x, y)
    }

    /// Creates a new 3D vector.
    pub const fn vec3<T: Scalar>(x: T, y: T, z: T) -> TVec3<T> {
        TVec3::new(x, y, z)
    }

    /// Creates a new 4D vector.
    pub const fn _vec4<T: Scalar>(x: T, y: T, z: T, w: T) -> TVec4<T> {
        TVec4::new(x, y, z, w)
    }
}

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
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
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

    pub unsafe fn destroy(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) };
    }
}

struct LoadedExtensions {
    synchronisation2: khr::synchronization2::Device,
    dynamic_rendering: khr::dynamic_rendering::Device,
}

impl LoadedExtensions {
    fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        let synchronisation2 = khr::synchronization2::Device::new(&instance, &device);
        let dynamic_rendering = khr::dynamic_rendering::Device::new(&instance, &device);
        Self { synchronisation2, dynamic_rendering }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    pos: glm::Vec2,
    color: glm::Vec3,
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, color) as u32),
        ]
    }
}

const VERTICES: [Vertex; 4] = [
    Vertex { pos: glm::vec2(-0.5, -0.5), color: glm::vec3(1., 0., 0.) },
    Vertex { pos: glm::vec2(0.5, -0.5), color: glm::vec3(0., 1., 0.) },
    Vertex { pos: glm::vec2(0.5, 0.5), color: glm::vec3(0., 0., 1.) },
    Vertex { pos: glm::vec2(-0.5, 0.5), color: glm::vec3(1., 1., 1.) },
];
const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    proj: glm::Mat4,
}

pub struct GalaxyEngine {
    _entry: ash::Entry,
    instance: ash::Instance,
    loaded_extensions: LoadedExtensions,
    debug_messenger: Option<DebugMessenger>,
    surface: Surface,
    device: Device,
    swapchain: Swapchain,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    graphics_cmd_pool: vk::CommandPool,
    transfer_cmd_pool: vk::CommandPool,
    uniform_buffers: [Buffer; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    uniform_buffers_mapped: [*mut u8; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    cmd_buffers: [vk::CommandBuffer; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    image_available_semaphores: [vk::Semaphore; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    render_finished_semaphores: [vk::Semaphore; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    in_flight_fences: [vk::Fence; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    current_frame: u32,
    start_time: std::time::Instant,
    window_size: vk::Extent2D,
    window_resized: bool,
}

impl GalaxyEngine {
    const MIN_VK_VERSION: u32 = vk::make_api_version(0, 1, 2, 0);
    const ENGINE_NAME: &'static CStr = c"Galaxy Engine";
    const ENGINE_VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");
    const MAX_FRAMES_IN_FLIGHT: usize = 2;

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
            .api_version(api_version);

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

        // Load extensions.
        let loaded_extensions = LoadedExtensions::new(&instance, device.device());

        // Create swapchain.
        let window_size = vk::Extent2D { width, height };
        let swapchain = Swapchain::new(&instance, &device, &surface, window_size, None)?;

        // Create graphics pipeline.

        // TODO: More robust shader file resolution.
        let vertex_shader_code = std::fs::read("galaxy_engine/shaders/shader.vert.spv")?;
        let fragment_shader_code = std::fs::read("galaxy_engine/shaders/shader.frag.spv")?;

        struct ShaderModule<'a> {
            module: vk::ShaderModule,
            stage: vk::ShaderStageFlags,
            _marker: std::marker::PhantomData<&'a ()>,
        }
        impl<'a> ShaderModule<'a> {
            fn new(device: &Device, code: &'a [u8], stage: vk::ShaderStageFlags) -> VkResult<Self> {
                let (prefix, code, suffix) = unsafe { code.align_to::<u32>() };
                assert!(prefix.is_empty());
                assert!(suffix.is_empty());
                let create_info = vk::ShaderModuleCreateInfo::default().code(code);
                Ok(Self { module: unsafe { device.device().create_shader_module(&create_info, None) }?, stage, _marker: std::marker::PhantomData })
            }
            fn get_stage_info(&self) -> vk::PipelineShaderStageCreateInfo {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(self.stage)
                    .module(self.module)
                    .name(c"main")
            }
            unsafe fn destroy(&mut self, device: &Device) {
                unsafe { device.device().destroy_shader_module(self.module, None) };
            }
        }

        let mut vertex_shader_module = ShaderModule::new(&device, &vertex_shader_code, vk::ShaderStageFlags::VERTEX)?;
        let mut fragment_shader_module = ShaderModule::new(&device, &fragment_shader_code, vk::ShaderStageFlags::FRAGMENT)?;

        let vertex_shader_stage_info = vertex_shader_module.get_stage_info();
        let fragment_shader_stage_info = fragment_shader_module.get_stage_info();
        let shader_stages = [vertex_shader_stage_info, fragment_shader_stage_info];

        // Vertex binding.
        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let transfer_cmd_pool = device.create_transient_command_pool(QueueFamily::Transfer)?;

        // Vertex buffer.
        let vertex_buffer = Buffer::new_for_typed_data(
            &device,
            &VERTICES,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        buffer::copy_via_staging_buffer(&device, transfer_cmd_pool, &VERTICES, &vertex_buffer)?;

        // Index buffer.
        let index_buffer = Buffer::new_for_typed_data(
            &device,
            &INDICES,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        buffer::copy_via_staging_buffer(&device, transfer_cmd_pool, &INDICES, &index_buffer)?;

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Create descriptor sets.
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(slice::from_ref(&ubo_layout_binding));
        let descriptor_set_layout = unsafe { device.device().create_descriptor_set_layout(&descriptor_set_layout_info, None) }?;

        // Create uniform buffers.
        let uniform_buffers: [Buffer; Self::MAX_FRAMES_IN_FLIGHT] = core::array::from_fn(|_| {
            Buffer::new(
                &device,
                std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).unwrap()
        });
        let uniform_buffers_mapped = core::array::from_fn(|i| {
            uniform_buffers[i].map(&device, 0, Some(std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize)).unwrap()
        });
        
        // Create descriptor pool.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(slice::from_ref(&pool_size))
            .max_sets(Self::MAX_FRAMES_IN_FLIGHT as u32);
        
        let descriptor_pool = unsafe { device.device().create_descriptor_pool(&descriptor_pool_info, None) }?;
        
        // Create descriptor sets.
        let layouts = [descriptor_set_layout; Self::MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.device().allocate_descriptor_sets(&alloc_info) }?;
        
        for (i ,descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i].handle())
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize);
            
            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(*descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&buffer_info));
            
            unsafe { device.device().update_descriptor_sets(slice::from_ref(&descriptor_write), &[]) };
        }

        let pipeline_dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_format = device_properties.swapchain_format.format;

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend_attachments = [color_blend_attachment];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe { device.device().create_pipeline_layout(&pipeline_layout_info, None) }?;

        let mut dynamic_pipeline_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&color_format));

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&pipeline_dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut dynamic_pipeline_info);

        let pipeline = unsafe { device.device().create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }.map_err(|(_, e)| e)?[0];

        // Drop shader modules after pipeline creation.
        unsafe { vertex_shader_module.destroy(&device) };
        unsafe { fragment_shader_module.destroy(&device) };

        // Create command pool.
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device_properties.graphics_queue_family_idx);
        let graphics_cmd_pool = unsafe { device.device().create_command_pool(&command_pool_info, None) }?;

        // Create command buffer.
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(graphics_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
        let command_buffers = unsafe { device.device().allocate_command_buffers(&command_buffer_info) }?;

        // Create sync objects.
        let mut image_available_semaphores = [vk::Semaphore::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut render_finished_semaphores = [vk::Semaphore::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let mut in_flight_fences = [vk::Fence::null(); Self::MAX_FRAMES_IN_FLIGHT];
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores[i] = unsafe { device.device().create_semaphore(&Default::default(), None) }?;
            render_finished_semaphores[i] = unsafe { device.device().create_semaphore(&Default::default(), None) }?;
            in_flight_fences[i] = unsafe { device.device().create_fence(&fence_info, None) }?;
        }

        Ok(Self {
            _entry: entry,
            instance,
            loaded_extensions,
            debug_messenger,
            surface,
            device,
            swapchain,
            vertex_buffer,
            index_buffer,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets: descriptor_sets.try_into().unwrap(),
            pipeline_layout,
            pipeline,
            graphics_cmd_pool,
            transfer_cmd_pool,
            uniform_buffers,
            uniform_buffers_mapped,
            cmd_buffers: command_buffers.try_into().unwrap(),
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            start_time: std::time::Instant::now(),
            window_size,
            window_resized: false,
        })
    }

    pub fn main_loop(&mut self) -> VkResult<()> {
        if self.window_resized {
            self.window_resized = false;
            self.recreate_swapchain()?;
        }

        let device = self.device.device();

        let sync2 = &self.loaded_extensions.synchronisation2;
        let dyn_cmd = &self.loaded_extensions.dynamic_rendering;

        let current_frame = self.current_frame as usize;

        // Update uniform buffer.
        let time = self.start_time.elapsed().as_secs_f32();
        let mut ubo = UniformBufferObject {
            model: nalgebra_glm::rotate(&glm::Mat4::identity(), time * 90f32.to_radians(), &glm::vec3(0., 0., 1.)),
            view: nalgebra_glm::look_at(&glm::vec3(2., 2., 2.), &glm::vec3(0., 0., 0.), &glm::vec3(0., 0., 1.)),
            proj: nalgebra_glm::perspective(self.window_size.width as f32 / self.window_size.height as f32, 45f32.to_radians(), 0.1, 10.0),
        };
        ubo.proj[(1,1)] *= -1.0;
        
        // Copy UBO to uniform buffer.
        let ubo_size = std::mem::size_of::<UniformBufferObject>();
        unsafe { std::ptr::copy_nonoverlapping(bytemuck::bytes_of(&ubo).as_ptr(), self.uniform_buffers_mapped[current_frame], ubo_size) };
        
        // Wait for fence.
        unsafe { device.wait_for_fences(&[self.in_flight_fences[current_frame]], true, u64::MAX) }?;

        // Acquire image from swapchain.
        let (image_idx, _is_suboptimal) = match self.swapchain.acquire_next_image(self.image_available_semaphores[current_frame], vk::Fence::null()) {
            Ok(x) => x,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return self.recreate_swapchain();
            }
            Err(_) => return Err(vk::Result::ERROR_UNKNOWN),
        };

        unsafe { device.reset_fences(&[self.in_flight_fences[current_frame]]) }?;

        let command_buffer = self.cmd_buffers[current_frame];

        unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()) }?;

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
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }?;

        let color_optimal_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::empty())
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(self.swapchain.get_subresource_range());

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&color_optimal_transition));
        unsafe { sync2.cmd_pipeline_barrier2(command_buffer, &dependency_info) };

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.get_image_views()[image_idx as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: swapchain_extent })
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment_info));

        unsafe { dyn_cmd.cmd_begin_rendering(command_buffer, &rendering_info) }
        unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline) };
        unsafe { device.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&self.vertex_buffer.handle()), slice::from_ref(&0)) };
        unsafe { device.cmd_bind_index_buffer(command_buffer, self.index_buffer.handle(), 0, vk::IndexType::UINT16) };
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, slice::from_ref(&self.descriptor_sets[current_frame]), &[]) };
        unsafe { device.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &[scissor]) };
        unsafe { device.cmd_draw_indexed(command_buffer, self.index_buffer.len(), 1, 0, 0, 0) };
        unsafe { dyn_cmd.cmd_end_rendering(command_buffer) };

        let color_optimal_to_present_src_transition = vk::ImageMemoryBarrier2::default()
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain.get_images()[image_idx as usize])
            .subresource_range(self.swapchain.get_subresource_range());

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&color_optimal_to_present_src_transition));
        unsafe { sync2.cmd_pipeline_barrier2(command_buffer, &dependency_info) };

        unsafe { device.end_command_buffer(command_buffer) }?;

        // Submit command buffer.
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(slice::from_ref(&command_buffer))
            .wait_semaphores(slice::from_ref(&self.image_available_semaphores[current_frame]))
            .wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
            .signal_semaphores(slice::from_ref(&self.render_finished_semaphores[current_frame]));

        unsafe { device.queue_submit(self.device.get_queue(QueueFamily::Graphics), slice::from_ref(&submit_info), self.in_flight_fences[current_frame]) }?;

        match self.swapchain.queue_present(self.device.get_queue(QueueFamily::Present), image_idx, &[self.render_finished_semaphores[current_frame]]) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain()?;
            }
            Err(e) => return Err(e),
        }

        self.current_frame = (self.current_frame + 1) % Self::MAX_FRAMES_IN_FLIGHT as u32;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> VkResult<()> {
        unsafe { self.device.device().device_wait_idle() }?;
        let new_swapchain = Swapchain::new(&self.instance, &self.device, &self.surface, self.window_size, Some(&self.swapchain))?;
        unsafe { self.swapchain.destroy(&self.device) };
        self.swapchain = new_swapchain;
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
        let device = self.device.device();

        unsafe { device.device_wait_idle() }.unwrap_or_else(|e| log::error!("Failed to wait for device idle: {:?}", e));

        // Drop sync objects.
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            unsafe { device.destroy_semaphore(self.image_available_semaphores[i], None) };
            unsafe { device.destroy_semaphore(self.render_finished_semaphores[i], None) };
            unsafe { device.destroy_fence(self.in_flight_fences[i], None) };
        }

        // Drop command_buffers.
        unsafe { device.free_command_buffers(self.graphics_cmd_pool, &self.cmd_buffers) };

        // Drop command_pools.
        unsafe { device.destroy_command_pool(self.graphics_cmd_pool, None) };
        unsafe { device.destroy_command_pool(self.transfer_cmd_pool, None) };

        // Drop pipeline.
        unsafe { device.destroy_pipeline(self.pipeline, None) };

        // Drop pipeline layout.
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };

        // Drop descriptor set layout.
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        // Drop descriptor pool.
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };

        // Drop vertex buffer.
        unsafe { self.vertex_buffer.destroy(&self.device) };
        // Drop index buffer.
        unsafe { self.index_buffer.destroy(&self.device) };
        // Drop uniform buffers.
        for uniform_buffer in self.uniform_buffers.iter_mut() {
            unsafe { uniform_buffer.destroy(&self.device) };
        }

        // Drop swapchain.
        unsafe { self.swapchain.destroy(&self.device) };

        // Drop device.
        unsafe { self.device.destroy() };

        // Drop surface.
        unsafe { self.surface.destroy() };

        // Drop debug messenger.
        if let Some(debug_messenger) = &mut self.debug_messenger {
            unsafe { debug_messenger.destroy() };
        }

        // Drop instance.
        unsafe { self.instance.destroy_instance(None) };

        // Entry is automatically dropped.
    }
}