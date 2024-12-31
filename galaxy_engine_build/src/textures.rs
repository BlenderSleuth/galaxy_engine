// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use galaxy_engine_config::load_ron_config;
use glob::glob;
use image::DynamicImage;
use serde::Deserialize;

use crate::{current_dir, rerun_if_changed, OutputDir, CONTENT_DIR};

#[derive(bincode::Encode, Deserialize, Debug, Copy, Clone)]
enum TextureComponents {
    Greyscale,
    #[serde(rename = "RG")]
    Rg,
    #[serde(rename = "RGB")]
    Rgb,
    #[serde(rename = "RGBA")]
    Rgba,
}

#[derive(bincode::Encode, Deserialize, Debug, Copy, Clone)]
enum TextureType {
    Colour(TextureComponents),
    Linear(TextureComponents),
    NormalFromBump,
    Normal,
}

fn default_mipmaps() -> bool {
    true
}

#[derive(bincode::Encode, Deserialize, Debug)]
struct Texture<'a> {
    path: &'a str,
    #[serde(rename = "type")]
    ty: TextureType,
    #[serde(default = "default_mipmaps")]
    mipmap: bool,
}

impl<'a> Texture<'a> {
    fn build(&self, config_path: &Path, filename: &str) {
        rerun_if_changed(config_path);

        // Set up input and output file.
        let built_filename = crate::str_with_extension(filename, "ktx2");
        let mut input_texture_path = config_path.with_file_name("");
        input_texture_path.push(self.path);
        let content_texture_path = input_texture_path.clone();
        rerun_if_changed(&input_texture_path);

        // Only build the texture if it doesn't already exist in the cache.
        if crate::cache::exists_in_cache(self, &content_texture_path, &built_filename, None) {
            crate::cache::copy_from_cache_to_build(&content_texture_path, &built_filename);
            return;
        }

        // Run ktx create. https://github.khronos.org/KTX-Software/ktxtools/ktx_create.html.
        let mut command = Command::new("ktx");
        command.arg("create").current_dir(current_dir());

        match self.ty {
            TextureType::Colour(dimensions) => {
                command
                    .args(["--assign-oetf", "sRGB"])
                    .args(["--assign-primaries", "sRGB"])
                    .args([
                        "--format",
                        match dimensions {
                            TextureComponents::Greyscale => "R8_SRGB",
                            TextureComponents::Rg => "R8G8_SRGB",
                            TextureComponents::Rgb => "R8G8B8_SRGB",
                            TextureComponents::Rgba => "R8G8B8A8_SRGB",
                        },
                    ]);
            }
            TextureType::Linear(dimensions) => {
                command.args(["--assign-oetf", "linear"]).args([
                    "--format",
                    match dimensions {
                        TextureComponents::Greyscale => "R8_UNORM",
                        TextureComponents::Rg => "R8G8_UNORM",
                        TextureComponents::Rgb => "R8G8B8_UNORM",
                        TextureComponents::Rgba => "R8G8B8A8_UNORM",
                    },
                ]);
            }
            TextureType::NormalFromBump => {
                // Load the bump texture.
                let bump_texture = image::open(&input_texture_path)
                    .expect("Failed to open bump texture")
                    .into_rgb8();

                let built_normal_filename = crate::str_with_extension(filename, "normal.png");

                let cached_normal_path = crate::convert_content_to_output_dir(
                    &content_texture_path,
                    &built_normal_filename,
                    OutputDir::Cache,
                )
                .unwrap();

                if !crate::cache::exists_in_cache(
                    self,
                    &content_texture_path,
                    &built_normal_filename,
                    Some(bump_texture.as_raw()),
                ) {
                    // Generate the normal texture.
                    let normal_texture =
                        normal_heights::map_normals_with_strength(&DynamicImage::ImageRgb8(bump_texture), 1.0);

                    // Save the normal texture to disk.
                    crate::create_required_folders(&cached_normal_path);
                    normal_texture
                        .save(&cached_normal_path)
                        .expect("Failed to save normal texture to disk");
                }

                input_texture_path = cached_normal_path;
                command
                    .args(["--assign-oetf", "srgb"])
                    .args(["--format", "R8G8B8_UNORM"])
                    .arg("--normal-mode");
            }
            TextureType::Normal => {
                command
                    //.args(["--assign-primaries", "BT709"])
                    .args(["--assign-oetf", "srgb"])
                    .args(["--format", "R8G8B8_UNORM"])
                    .arg("--normal-mode");
            }
        };

        // Set up UASTC compression.
        command.args(["--encode", "uastc", "--uastc-rdo"]);

        if cfg!(feature = "packaged") {
            command.args(["--uastc-quality", "5"]).args(["--zstd", "14"]);
        } else {
            command.args(["--uastc-quality", "2"]).args(["--zstd", "10"]);
        }

        if self.mipmap {
            command.arg("--generate-mipmap");
        } else {
            unimplemented!("Mipmaps must be generated for now.");
        }

        // Create output paths.
        let output_file_path =
            crate::convert_content_to_output_dir(config_path, &built_filename, OutputDir::Cache).unwrap();
        crate::create_required_folders(&output_file_path);

        command.arg(&input_texture_path).arg(&output_file_path);

        crate::handle_command_result(command.output(), "Texture build failed", "ktx");

        // Move the texture from the cache to the build folder.
        crate::cache::copy_from_cache_to_build(&content_texture_path, &built_filename);
    }
}

pub fn build_textures(glob_texture_paths: &[&str]) {
    for glob_path in glob_texture_paths {
        for path in glob(&crate::str_path_join(CONTENT_DIR, glob_path)).expect("Failed to read texture glob pattern") {
            let config_path = path.unwrap();
            rerun_if_changed(&config_path);

            let texture_config_source =
                std::fs::read_to_string(&config_path).expect("Failed to read texture config file.");
            let configs: HashMap<&str, Texture> =
                load_ron_config(&texture_config_source).expect("Failed to load texture config.");
            for (filename, config) in configs {
                config.build(&config_path, filename);
            }
        }
    }
}
