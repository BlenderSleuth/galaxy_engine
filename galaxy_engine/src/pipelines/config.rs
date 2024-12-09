// Copyright (c) 2024 Ben Sutherland.

use std::hash::Hash;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;

use crate::level::DrawData;
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
pub(super) struct RasteriserConfig {
    pub multisample_enable: bool,
    pub depth_enable: bool,
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

#[derive(serde::Deserialize, Debug, Copy, Clone)]
pub enum PipelineBindingDataSize {
    Float,
    Float2,
    Float3,
    Float4,
    Normal,
}

impl PipelineBindingDataSize {
    const FLOAT_SIZE: usize = std::mem::size_of::<f32>();

    pub const fn layout(&self) -> std::alloc::Layout {
        match std::alloc::Layout::from_size_align(self.size(), self.align()) {
            Ok(layout) => layout,
            Err(_) => panic!("Alignment must be a power of 2."),
        }
    }

    pub const fn size(&self) -> usize {
        Self::FLOAT_SIZE
            * match self {
                Self::Float => 1,
                Self::Float2 => 2,
                Self::Float3 | Self::Normal => 3,
                Self::Float4 => 4,
            }
    }

    pub const fn align(&self) -> usize {
        Self::FLOAT_SIZE
            * match self {
                Self::Float => 1,
                Self::Float2 => 2,
                Self::Float3 | Self::Normal => 4, // Float3 uses Float4 (16 byte) alignment.
                Self::Float4 => 4,
            }
    }
}

#[derive(serde::Deserialize, Debug, Copy, Clone)]
pub struct PipelineBinding {
    #[serde(rename = "type")]
    pub ty: PipelineBindingDataSize,
    // TODO: These are not currently used.
    pub stages: GraphicsShaderStageFlags,
}

pub type PipelineBindingMap<S = Arc<str>> = IndexMap<S, PipelineBinding>;

#[derive(serde::Deserialize, Debug, Hash, Copy, Clone, PartialEq, Eq)]
pub enum PushConstantBinding {
    DrawData,
}

impl PushConstantBinding {
    pub fn push_constant_range(&self) -> vk::PushConstantRange {
        match self {
            Self::DrawData => vk::PushConstantRange::default()
                .stage_flags((GraphicsShaderStageFlags::Vertex | GraphicsShaderStageFlags::Fragment).vk())
                .offset(0)
                .size(std::mem::size_of::<DrawData>() as u32),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub(crate) struct PipelineLayoutNamedBindings<'a> {
    pub push_constant: Option<PushConstantBinding>,
    #[serde(borrow)]
    pub bindings: PipelineBindingMap<&'a str>,
}

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

#[derive(serde::Deserialize, Debug)]
pub(super) struct GraphicsPipelineConfig<'a> {
    pub name: Arc<str>,
    #[serde(borrow)]
    pub shaders: GraphicsShaderConfig<'a>,
    pub rasteriser: RasteriserConfig,
    #[serde(borrow)]
    pub layout: PipelineLayoutNamedBindings<'a>,
}

#[derive(serde::Deserialize, Debug)]
pub(super) struct ComputePipelineConfig<'a> {
    name: &'a str,
    shader: &'a str,
    #[serde(borrow)]
    layout: PipelineLayoutNamedBindings<'a>,
}

#[derive(serde::Deserialize, Debug)]
pub(crate) enum PipelineConfig<'a> {
    #[serde(borrow, rename = "GraphicsPipeline")]
    Graphics(GraphicsPipelineConfig<'a>),
    #[serde(borrow, rename = "ComputePipeline")]
    Compute(ComputePipelineConfig<'a>),
}
