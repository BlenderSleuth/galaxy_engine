// Copyright (c) 2024-2025 Ben Sutherland.

mod config;
mod material;
mod material_manager;

pub use material::{Material, MaterialError};
pub use material_manager::LoadingMaterialManager;
pub(crate) use material_manager::MaterialManager;

use crate::materials::config::ResourceConstant;

// A shader reference to a texture or constant resource. The top bit is set to indicate a texture.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct ResourceRef(u32);

impl ResourceRef {
    const TEXTURE_BIT: u32 = 0x80000000;
    fn texture(index: u32) -> Self {
        assert!(index < Self::TEXTURE_BIT, "Texture index out of range.");
        Self(index | Self::TEXTURE_BIT)
    }
    fn constant(index: u32) -> Self {
        assert!(index < Self::TEXTURE_BIT, "Constant index out of range.");
        Self(index)
    }
}

pub enum ResourceBinding {
    Texture(u32),
    Constant(ResourceConstant),
}
