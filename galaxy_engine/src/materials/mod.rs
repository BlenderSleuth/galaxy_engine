// Copyright (c) 2024 Ben Sutherland.

mod config;
mod material;
mod material_manager;

pub use material::{Material, MaterialError, ResourceBinding, ResourceBindingMap};
pub use material_manager::LoadingMaterialManager;
pub(crate) use material_manager::MaterialManager;
