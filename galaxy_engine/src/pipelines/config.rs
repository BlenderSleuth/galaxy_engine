// Copyright (c) 2024-2025 Ben Sutherland.

use std::hash::Hash;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use ultraviolet::Vec2;

use crate::vertex_input::VertexInputType;

#[derive(serde::Deserialize, Debug)]
pub(super) struct VertexShaderConfig<'a> {
    pub id: &'a str,
    pub input_type: VertexInputType,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct FragmentShaderConfig<'a> {
    pub id: &'a str,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct GraphicsShaderConfig<'a> {
    #[serde(borrow)]
    pub vertex: VertexShaderConfig<'a>,
    #[serde(borrow)]
    pub fragment: FragmentShaderConfig<'a>,
}

#[derive(serde::Deserialize, Debug)]
pub enum BackfaceCullMode {
    None,
    Front,
    Back,
}

impl BackfaceCullMode {
    pub fn vk(&self) -> vk::CullModeFlags {
        match self {
            Self::None => vk::CullModeFlags::NONE,
            Self::Front => vk::CullModeFlags::FRONT,
            Self::Back => vk::CullModeFlags::BACK,
        }
    }
}

impl Default for BackfaceCullMode {
    fn default() -> Self {
        Self::Back
    }
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct RasteriserConfig {
    pub multisample_enable: bool,
    pub depth_enable: bool,
    #[serde(default)]
    pub backface_cull_mode: BackfaceCullMode,
    #[serde(default)]
    pub transparent: bool,
}

//bitflags::bitflags! {
//    #[derive(serde::Serialize, serde::Deserialize, Debug, Hash, PartialEq, Eq, Copy, Clone)]
//    #[serde(transparent)]
//    pub struct GraphicsShaderStageFlags: u8 {
//        const Vertex = 1;
//        const Fragment = 2;
//    }
//}
//
//impl GraphicsShaderStageFlags {
//    pub fn vk(&self) -> vk::ShaderStageFlags {
//        let mut flags = vk::ShaderStageFlags::empty();
//        if self.contains(Self::Vertex) {
//            flags |= vk::ShaderStageFlags::VERTEX;
//        }
//        if self.contains(Self::Vertex) {
//            flags |= vk::ShaderStageFlags::FRAGMENT;
//        }
//        flags
//    }
//}

//#[derive(serde::Deserialize, Debug, Copy, Clone)]
//pub struct GraphicsPipelineBinding {
//    #[serde(rename = "type")]
//    pub ty: PipelineBindingDataSize,
//    pub stages: GraphicsShaderStageFlags,
//}

#[derive(serde::Deserialize, Debug, Copy, Clone)]
pub enum PipelineBindingDataSize {
    Float,
    Float2,
    Float3,
    Float4,
    Normal,
}

impl PipelineBindingDataSize {
    pub const fn len(&self) -> usize {
        match self {
            Self::Float => 1,
            Self::Float2 => 2,
            Self::Float3 | Self::Normal => 3,
            Self::Float4 => 4,
        }
    }
}

pub type PipelineBindingMap<S = Arc<str>> = IndexMap<S, PipelineBindingDataSize>;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GuiPushConstants {
    screen_size: Vec2,
    tex_idx: u32,
}

// Todo: better push constant management. List of sizes and shader stages.
#[derive(serde::Deserialize, Debug, Hash, Copy, Clone, PartialEq, Eq)]
pub enum PushConstantBinding {
    DrawOffset,
    //Gui,
    ComputeInt,
    ComputeInt2,
    ComputeInt4,
}

impl PushConstantBinding {
    //const DRAW_OFFSET: vk::PushConstantRange = vk::PushConstantRange {
    //    stage_flags: vk::ShaderStageFlags::VERTEX,
    //    offset: 0,
    //    size: std::mem::size_of::<u32>() as u32,
    //};
    //const GUI: [vk::PushConstantRange; 2] = [
    //    vk::PushConstantRange {
    //        stage_flags: vk::ShaderStageFlags::VERTEX,
    //        offset: std::mem::offset_of!(GuiPushConstants, screen_size) as u32,
    //        size: std::mem::size_of::<Vec2>() as u32,
    //    },
    //    vk::PushConstantRange {
    //        stage_flags: vk::ShaderStageFlags(vk::ShaderStageFlags::FRAGMENT.0 | vk::ShaderStageFlags::VERTEX.0),
    //        offset: std::mem::offset_of!(GuiPushConstants, tex_idx) as u32,
    //        size: std::mem::size_of::<u32>() as u32,
    //    },
    //];
    //const COMPUTE_INT: vk::PushConstantRange = vk::PushConstantRange {
    //    stage_flags: vk::ShaderStageFlags::COMPUTE,
    //    offset: 0,
    //    size: std::mem::size_of::<u32>() as u32,
    //};
    //const COMPUTE_INT2: vk::PushConstantRange = vk::PushConstantRange {
    //    stage_flags: vk::ShaderStageFlags::COMPUTE,
    //    offset: 0,
    //    size: std::mem::size_of::<u32>() as u32 * 2,
    //};
    //const COMPUTE_INT4: vk::PushConstantRange = vk::PushConstantRange {
    //    stage_flags: vk::ShaderStageFlags::COMPUTE,
    //    offset: 0,
    //    size: std::mem::size_of::<u32>() as u32 * 4,
    //};

    pub fn push_constant_ranges(&self) -> vk::PushConstantRange {
        // Note: Generally only access push constant in one shader stage.
        // TODO: Once graphics pipelines are sorted out properly, use draw offset correctly (currently set up for GUI).
        match self {
            Self::DrawOffset => vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(size_of::<u32>() as u32 * 3),
            Self::ComputeInt => vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<u32>() as u32),
            Self::ComputeInt2 => vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<u32>() as u32 * 2),
            Self::ComputeInt4 => vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<u32>() as u32 * 4),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub(crate) struct GraphicsPipelineLayoutBindings<'a> {
    pub push_constant: Option<PushConstantBinding>,
    #[serde(borrow)]
    pub bindings: PipelineBindingMap<&'a str>,
}

//impl GraphicsPipelineLayoutBindings<'_> {
//    pub fn push_constant_range(&self) -> Option<vk::PushConstantRange> {
//        self.push_constant.map(|binding| binding.push_constant_range())
//    }
//}

//impl PipelineLayoutNamedBindings {
//    pub fn bindings(&self) -> PipelineLayoutBindings {
//        let bindings = self.bindings.values().copied().collect();
//        PipelineLayoutBindings {
//            push_constant: self.push_constant,
//            bindings,
//        }
//    }
//
//    pub fn push_constant(&self) -> Option<PipelineLayoutBinding> {
//        self.push_constant
//    }
//}

//const NUM_DESCRIPTOR_BINDINGS: usize = 12;

//#[derive(Hash, Debug, PartialEq, Eq)]
//pub(crate) struct PipelineLayoutBindings {
//    pub push_constant: Option<PipelineLayoutBinding>,
//    pub bindings: ArrayVec<PipelineLayoutBinding, NUM_DESCRIPTOR_BINDINGS>,
//}

//#[derive(serde::Deserialize, Debug)]
//pub enum GraphicsRenderPhase {
//    Opaque,
//    AlphaClip,
//}

#[derive(serde::Deserialize, Debug)]
pub(super) struct GraphicsPipelineConfig<'a> {
    #[serde(skip)]
    pub id: Arc<str>,
    #[serde(borrow)]
    pub shaders: GraphicsShaderConfig<'a>,
    //pub phase: GraphicsRenderPhase,
    pub rasteriser: RasteriserConfig,
    #[serde(borrow)]
    pub layout: GraphicsPipelineLayoutBindings<'a>,
}

#[derive(serde::Deserialize, Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum ComputeResourceType {
    UniformBuffer,
    StorageBuffer,
}

impl ComputeResourceType {
    pub fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub(crate) struct ComputePipelineLayoutBindings<'a> {
    pub push_constant: Option<PushConstantBinding>,
    #[serde(borrow)]
    pub bindings: IndexMap<&'a str, ComputeResourceType>,
}

impl ComputePipelineLayoutBindings<'_> {
    //pub fn push_constant_range(&self) -> Option<vk::PushConstantRange> {
    //    self.push_constant.map(|binding| binding.push_constant_range())
    //}

    pub fn binding_types(&self) -> Vec<ComputeResourceType> {
        self.bindings.values().copied().collect()
    }
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct ComputePipelineConfig<'a> {
    #[serde(skip)]
    pub id: Arc<str>,
    pub shader: &'a str,
    pub num_threads: [u32; 3],
    #[serde(borrow)]
    pub layout: ComputePipelineLayoutBindings<'a>,
}

#[derive(serde::Deserialize, Debug)]
pub(crate) enum PipelineConfig<'a> {
    #[serde(borrow, rename = "GraphicsPipeline")]
    Graphics(GraphicsPipelineConfig<'a>),
    #[serde(borrow, rename = "ComputePipeline")]
    Compute(ComputePipelineConfig<'a>),
}

impl PipelineConfig<'_> {
    pub fn with_id(mut self, id: &Arc<str>) -> Self {
        let id = Arc::clone(id);
        match &mut self {
            Self::Graphics(config) => config.id = id,
            Self::Compute(config) => config.id = id,
        }
        self
    }
}
