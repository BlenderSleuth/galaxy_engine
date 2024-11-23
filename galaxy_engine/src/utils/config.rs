// Copyright (c) 2024 Ben Sutherland.

use arrayvec::ArrayString;

#[derive(thiserror::Error, Debug)]
pub(crate) enum ConfigLoadError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
}

// Same size as a standard string, but limited and stack allocated.
pub(crate) type ConfigID = ArrayString<20>;
static_assertions::assert_eq_size!(ConfigID, String);
