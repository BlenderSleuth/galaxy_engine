// Copyright (c) 2024-2025 Ben Sutherland.

pub use galaxy_engine_config::*;

#[derive(thiserror::Error, Debug)]
pub enum ConfigLoadError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
}
