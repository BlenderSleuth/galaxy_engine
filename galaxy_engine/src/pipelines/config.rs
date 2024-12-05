// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;
use indexmap::IndexMap;

use crate::level::DrawData;
use crate::pipelines::PipelineLayout;
use crate::vertex_input::VertexInputType;
use crate::vulkan::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};

#[derive(serde::Deserialize, Debug)]
pub(super) struct VertexShaderConfig {
    pub id: String,
    pub input_type: VertexInputType,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct FragmentShaderConfig {
    pub id: String,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct GraphicsShaderConfig {
    pub vertex: VertexShaderConfig,
    pub fragment: FragmentShaderConfig,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct RasteriserConfig {
    pub multisample_enable: bool,
    pub depth_enable: bool,
}

#[derive(serde::Deserialize, Debug, Hash, Copy, Clone, PartialEq, Eq)]
pub(super) enum PipelineLayoutDataType {
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    UInt,
    DrawData,
}

impl PipelineLayoutDataType {
    pub fn size(&self) -> u32 {
        (match self {
            Self::Float => std::mem::size_of::<f32>(),
            Self::Float2 => std::mem::size_of::<[f32; 2]>(),
            Self::Float3 => std::mem::size_of::<[f32; 4]>(),
            Self::Float4 => std::mem::size_of::<[f32; 4]>(),
            Self::Int => std::mem::size_of::<i32>(),
            Self::UInt => std::mem::size_of::<u32>(),
            Self::DrawData => std::mem::size_of::<DrawData>(),
        }) as u32
    }
    //pub fn align(&self) -> u32 {
    //    (match self {
    //        Self::Float => std::mem::align_of::<f32>(),
    //        Self::Float2 => std::mem::align_of::<[f32; 2]>(),
    //        Self::Float3 => std::mem::align_of::<[f32; 4]>(),
    //        Self::Float4 => std::mem::align_of::<[f32; 4]>(),
    //        Self::Int => std::mem::align_of::<i32>(),
    //        Self::UInt => std::mem::align_of::<u32>(),
    //        Self::DrawData => std::mem::align_of::<DrawData>(),
    //    }) as u32
    //}
}

bitflags::bitflags! {
    #[derive(serde::Serialize, serde::Deserialize, Debug, Hash, PartialEq, Eq, Copy, Clone)]
    #[serde(transparent)]
    pub struct GraphicsShaderStageFlags: u8 {
        const Vertex = 1;
        const Fragment = 2;
    }
}

impl GraphicsShaderStageFlags {
    pub fn vk(&self) -> vk::ShaderStageFlags {
        let mut flags = vk::ShaderStageFlags::empty();
        if self.contains(Self::Vertex) {
            flags |= vk::ShaderStageFlags::VERTEX;
        }
        if self.contains(Self::Vertex) {
            flags |= vk::ShaderStageFlags::FRAGMENT;
        }
        flags
    }
}

#[derive(serde::Deserialize, Hash, Debug, Copy, Clone, PartialEq, Eq)]
pub(super) struct PipelineLayoutBinding {
    #[serde(rename = "type")]
    pub ty: PipelineLayoutDataType,
    pub stages: GraphicsShaderStageFlags,
}

impl PipelineLayoutBinding {
    pub fn push_constant_range(&self) -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(self.stages.vk())
            .offset(0)
            .size(self.ty.size())
    }
}

#[derive(serde::Deserialize, Debug, Default)]
pub(crate) struct PipelineLayoutNamedBindings {
    push_constant: Option<PipelineLayoutBinding>,
    bindings: IndexMap<String, PipelineLayoutBinding>,
}

impl PipelineLayoutNamedBindings {
    pub fn bindings(&self) -> PipelineLayoutBindings {
        let bindings = self.bindings.values().copied().collect();
        PipelineLayoutBindings {
            push_constant: self.push_constant,
            bindings,
        }
    }
}

const NUM_DESCRIPTOR_BINDINGS: usize = 12;

#[derive(Hash, Debug, PartialEq, Eq)]
pub(crate) struct PipelineLayoutBindings {
    pub push_constant: Option<PipelineLayoutBinding>,
    pub bindings: ArrayVec<PipelineLayoutBinding, NUM_DESCRIPTOR_BINDINGS>,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct GraphicsPipelineConfig {
    pub name: String,
    pub shaders: GraphicsShaderConfig,
    pub rasteriser: RasteriserConfig,
    #[serde(default)]
    pub layout: PipelineLayoutNamedBindings,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct ComputePipelineConfig {
    name: String,
    shader: String,
    layout: PipelineLayoutNamedBindings,
}

#[derive(serde::Deserialize, Debug)]
pub(crate) enum PipelineConfig {
    #[serde(rename = "GraphicsPipeline")]
    Graphics(GraphicsPipelineConfig),
    #[serde(rename = "ComputePipeline")]
    Compute(ComputePipelineConfig),
}

pub(super) fn load_config(config_str: &str) -> ron::error::SpannedResult<PipelineConfig> {
    crate::utils::load_config(config_str)
}
pub type PipelineLayoutCache = HashMap<Option<PipelineLayoutBinding>, Arc<PipelineLayout>>;
pub type VertexShaderModuleCache = HashMap<String, ShaderModule<VertexShaderStage>>;
pub type FragmentShaderModuleCache = HashMap<String, ShaderModule<FragmentShaderStage>>;
