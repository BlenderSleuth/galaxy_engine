use ash::prelude::VkResult;
use ash::vk;
use crate::device::{Device, SharedDeviceLoader};

// Shader stage type state pattern.
pub trait ShaderStageType {
    fn stage() -> vk::ShaderStageFlags;
}
pub struct VertexShaderStage;
impl ShaderStageType for VertexShaderStage {
    fn stage() -> vk::ShaderStageFlags { vk::ShaderStageFlags::VERTEX }
}
pub struct FragmentShaderStage;
impl ShaderStageType for FragmentShaderStage {
    fn stage() -> vk::ShaderStageFlags { vk::ShaderStageFlags::FRAGMENT }
}
pub struct ComputeShaderStage;
impl ShaderStageType for ComputeShaderStage {
    fn stage() -> vk::ShaderStageFlags { vk::ShaderStageFlags::COMPUTE }
}

pub struct ShaderModule<S: ShaderStageType> {
    loader: SharedDeviceLoader,
    handle: vk::ShaderModule,
    _marker: std::marker::PhantomData<S>,
}

impl<S: ShaderStageType> ShaderModule<S> {
    pub fn new(device: &Device, code: &[u8]) -> VkResult<Self> {
        let (prefix, code, suffix) = unsafe { code.align_to::<u32>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        Ok(Self {
            loader: device.cloned_loader(),
            handle: unsafe { device.loader().create_shader_module(&create_info, None) }?,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn stage_info(&self) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage())
            .module(self.handle)
            .name(match self.stage() {
                vk::ShaderStageFlags::VERTEX => c"mainVS",
                vk::ShaderStageFlags::FRAGMENT => c"mainFS",
                vk::ShaderStageFlags::COMPUTE => c"mainCS",
                _ => panic!("Unsupported shader stage: {:?}", self.stage()),
            })
    }

    pub fn stage(&self) -> vk::ShaderStageFlags {
        S::stage()
    }
}

impl<S: ShaderStageType> Drop for ShaderModule<S> {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_shader_module(self.handle, None) };
    }
}
