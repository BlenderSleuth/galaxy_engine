// Copyright (c) 2024 Ben Sutherland.

pub mod config {
    pub fn load_config<'de, T: serde::Deserialize<'de>>(config_str: &'de str) -> ron::error::SpannedResult<T> {
        use ron::extensions::Extensions;
        ron::Options::default()
            .with_default_extension(
                Extensions::UNWRAP_VARIANT_NEWTYPES | Extensions::IMPLICIT_SOME | Extensions::UNWRAP_NEWTYPES,
            )
            .from_str(config_str)
    }
}
