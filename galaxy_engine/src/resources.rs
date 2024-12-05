// Copyright (c) 2024 Ben Sutherland.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::engine::GalaxyEngine;

pub trait ResourceType {
    const EXTENSION: &'static str;
}

pub mod resource_type {
    use super::ResourceType;

    pub enum Mesh {}
    impl ResourceType for Mesh {
        const EXTENSION: &'static str = "obj";
    }
    pub enum Texture {}
    impl ResourceType for Texture {
        const EXTENSION: &'static str = "ktx2";
    }
    pub enum Material {}
    impl ResourceType for Material {
        const EXTENSION: &'static str = "mat.ron";
    }
}

#[derive(Clone, Copy)]
pub enum ResourcePathBase {
    Game,
    Engine,
}

impl ResourcePathBase {
    pub fn new(path: &Path) -> Option<(Self, &Path)> {
        if let Ok(path) = path.strip_prefix("/engine/") {
            Some((Self::Engine, path))
        } else if let Ok(path) = path.strip_prefix("/game/") {
            Some((Self::Game, path))
        } else {
            None
        }
    }
}

pub type MeshResourcePath = ResourcePath<resource_type::Mesh>;
pub type TextureResourcePath = ResourcePath<resource_type::Texture>;
pub type MaterialResourcePath = ResourcePath<resource_type::Material>;

pub struct ResourcePath<R: ResourceType> {
    base: ResourcePathBase,
    path: PathBuf,
    resource_type: PhantomData<R>,
}

impl<R: ResourceType> ResourcePath<R> {
    pub fn new(path: &str) -> Option<Self> {
        ResourcePath::new_from_path(Path::new(path))
    }

    pub fn new_from_path(path: &Path) -> Option<Self> {
        let (base, path) = ResourcePathBase::new(path)?;
        Some(Self {
            base,
            path: path.to_path_buf(),
            resource_type: PhantomData,
        })
    }

    pub fn relative_resource<R2: ResourceType>(&self, relative_path: &str) -> ResourcePath<R2> {
        ResourcePath {
            base: self.base,
            path: self.path.with_file_name(relative_path),
            resource_type: PhantomData,
        }
    }

    pub fn full_path(&self, _engine: &GalaxyEngine) -> PathBuf {
        #[cfg(not(feature = "packaged"))]
        let mut base = match self.base {
            ResourcePathBase::Game => _engine.game_dir().join(&self.path),
            ResourcePathBase::Engine => Path::new(GalaxyEngine::CONTENT_DIR).join(&self.path),
        };

        // On packaged builds, everything is in a subfolder of the packaged folder.
        #[cfg(feature = "packaged")]
        let mut base = match self.base {
            ResourcePathBase::Game => Path::new(GalaxyEngine::CONTENT_DIR).join("engine").join(&self.path),
            ResourcePathBase::Engine => Path::new(GalaxyEngine::CONTENT_DIR).join("game").join(&self.path),
        };

        base.set_extension(R::EXTENSION);
        base
    }
}
