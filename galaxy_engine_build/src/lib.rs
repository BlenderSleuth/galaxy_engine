// Copyright (c) 2024-2025 Ben Sutherland.

mod cache;
mod shaders;
mod textures;

use std::path::{Path, PathBuf};
use std::process::Output;

use const_format::concatcp;
pub use shaders::compile_shaders;
pub use textures::build_textures;

const CONTENT_DIR: &str = "content/";
fn content_dir() -> &'static Path {
    Path::new(CONTENT_DIR)
}
const BUILD_TYPE_SUBDIR: &str = const {
    if cfg!(feature = "packaged") {
        "packaged/"
    } else {
        "debug/"
    }
};
const CACHE_DIR: &str = concatcp!("cache/", BUILD_TYPE_SUBDIR);
fn cache_dir() -> &'static Path {
    Path::new(CACHE_DIR)
}
const BUILD_DIR: &str = "build/";
fn build_dir() -> &'static Path {
    Path::new(BUILD_DIR)
}

enum OutputDir {
    Cache,
    Build,
}

impl OutputDir {
    fn to_path(&self) -> &'static Path {
        match self {
            OutputDir::Cache => cache_dir(),
            OutputDir::Build => build_dir(),
        }
    }
}

fn convert_content_to_output_dir(content_path: &Path, built_filename: &str, output_dir: OutputDir) -> Option<PathBuf> {
    Some(
        output_dir
            .to_path()
            .join(content_path.strip_prefix(content_dir()).ok()?)
            .with_file_name(built_filename),
    )
}

//fn convert_cache_to_build_dir(cache_path: &Path) -> Option<PathBuf> {
//    Some(build_dir().join(cache_path.strip_prefix(cache_dir()).ok()?))
//}

fn current_dir() -> String {
    std::env::var("CARGO_MANIFEST_DIR").unwrap()
}

fn str_path_join(a: &str, b: &str) -> String {
    // Given both a and b are valid strings, this should never fail.
    Path::new(a).join(b).into_os_string().into_string().unwrap()
}

fn str_with_extension(path: &str, extension: &str) -> String {
    Path::new(path)
        .with_extension(extension)
        .into_os_string()
        .into_string()
        .unwrap()
}

fn full_path(path: &Path) -> PathBuf {
    let mut full_path = PathBuf::from(current_dir());
    full_path.push(path);
    full_path
}

//fn core_filename(path: &Path) -> Option<&str> {
//    // Can use Path::file_prefix() when it's stable.
//    path.file_stem()?.to_str()?.split(".").next()
//}

fn create_required_folders(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
}

fn rerun_if_changed(path: &Path) {
    println!("cargo::rerun-if-changed={}", full_path(path).to_str().unwrap());
}

fn handle_command_result(cmd_result: std::io::Result<Output>, fail: &str, binary: &str) {
    match cmd_result {
        Ok(output) => {
            if !output.stderr.is_empty() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                for line in stderr.lines() {
                    print_warning!("{line}");
                }
                panic!("{fail}.");
            }
            if !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    print_warning!("{line}");
                }
            }
        }
        Err(err) => {
            print_warning!("{err}");
            panic!("{fail}. Make sure you have {binary} installed.");
        }
    };
}

#[macro_export]
macro_rules! print_warning {
    ($($tokens: tt)*) => {
        println!("cargo::warning={}", format!($($tokens)*))
    }
}
