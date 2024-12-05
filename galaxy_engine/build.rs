// Copyright (c) 2024 Ben Sutherland.

fn main() {
    // Compile shaders.
    galaxy_engine_build::compile_shaders(&["shaders/**/*.slang"], cfg!(feature = "debug_info"));
}
