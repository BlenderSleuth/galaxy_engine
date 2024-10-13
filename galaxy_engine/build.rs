fn main() {
    galaxy_engine_build::compile_shaders(&["shaders/*.hlsl"], cfg!(feature = "debug_info"));
}