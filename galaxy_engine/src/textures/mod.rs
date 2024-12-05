// Copyright (c) 2024 Ben Sutherland.

mod texture;
mod texture_manager;

pub use texture::{Texture, TextureError};
pub(crate) use texture_manager::{TextureIndex, TextureManager};
