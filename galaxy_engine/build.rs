// Copyright (c) 2024-2025 Ben Sutherland.

fn main() {
    // Compile shaders.
    galaxy_engine_build::compile_shaders(&["shaders/**/*.slang"]);
}
