// Copyright (c) 2024 Ben Sutherland.

use std::path::{Path, PathBuf};

use crate::engine::GalaxyEngine;

pub trait ResourceType {
    const EXTENSION: &'static str;
    const BUILT: bool;
}

pub mod resource_type {
    use super::ResourceType;

    pub enum Mesh {}
    impl ResourceType for Mesh {
        const EXTENSION: &'static str = "obj";
        const BUILT: bool = false;
    }
    pub enum Texture {}
    impl ResourceType for Texture {
        const EXTENSION: &'static str = "ktx2";
        const BUILT: bool = true;
    }
    pub enum Material {}
    impl ResourceType for Material {
        const EXTENSION: &'static str = "mat.ron";
        const BUILT: bool = false;
    }
    pub enum Level {}
    impl ResourceType for Level {
        const EXTENSION: &'static str = "level.ron";
        const BUILT: bool = false;
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
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

pub trait ResourcePathTrait {
    fn full_path(&self, engine: &GalaxyEngine) -> PathBuf;
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct ResourcePath {
    base: ResourcePathBase,
    path: PathBuf,
}

impl ResourcePath {
    pub fn new<P: AsRef<Path>>(path: P, relative_to: Option<&ResourcePath>) -> Option<Self> {
        if let Some((base, path)) = ResourcePathBase::new(path.as_ref()) {
            Some(Self {
                base,
                path: path.to_path_buf(),
            })
        } else {
            relative_to.map(|relative_to| Self {
                base: relative_to.base,
                path: relative_to.path.parent().unwrap_or(&relative_to.path).join(path),
            })
        }
    }

    pub fn full_path<R: ResourceType>(&self, engine: &GalaxyEngine) -> PathBuf {
        let mut base = match self.base {
            ResourcePathBase::Game => {
                if R::BUILT {
                    engine.game_dir().join(GalaxyEngine::BUILD_DIR).join(&self.path)
                } else {
                    engine.game_dir().join(GalaxyEngine::CONTENT_DIR).join(&self.path)
                }
            }
            ResourcePathBase::Engine => {
                if R::BUILT {
                    Path::new(GalaxyEngine::BUILT_PATH).join(&self.path)
                } else {
                    Path::new(GalaxyEngine::CONTENT_PATH).join(&self.path)
                }
            }
        };
        base.set_extension(R::EXTENSION);
        base
    }
}
