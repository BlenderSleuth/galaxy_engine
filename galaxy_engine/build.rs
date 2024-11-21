// Copyright (c) 2024 Ben Sutherland.

fn main() {
    galaxy_engine_build::compile_shaders(&["content/shaders/**/*.slang"], cfg!(feature = "debug_info"));
}
