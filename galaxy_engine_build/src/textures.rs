// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use galaxy_engine_utils::config::load_config;
use glob::glob;
use image::DynamicImage;
use serde::Deserialize;
use sha3::Digest;

use crate::{current_dir, rerun_if_changed, OutputDir, CONTENT_DIR};

#[derive(Deserialize, Debug, Copy, Clone)]
enum TextureComponents {
    Greyscale,
    #[serde(rename = "RG")]
    Rg,
    #[serde(rename = "RGB")]
    Rgb,
    #[serde(rename = "RGBA")]
    Rgba,
}

#[derive(Deserialize, Debug, Copy, Clone)]
enum TextureType {
    Colour(TextureComponents),
    Linear(TextureComponents),
    NormalFromBump,
    Normal,
}

fn default_mipmaps() -> bool {
    true
}

#[derive(Deserialize, Debug)]
struct Texture<'a> {
    path: &'a str,
    #[serde(rename = "type")]
    ty: TextureType,
    #[serde(default = "default_mipmaps")]
    mipmap: bool,
}

impl<'a> Texture<'a> {
    fn build(&self, config_path: &Path, filename: &str, debug: bool) {
        rerun_if_changed(config_path);

        // Run ktx create. https://github.khronos.org/KTX-Software/ktxtools/ktx_create.html.
        let mut command = Command::new("ktx");
        command.arg("create").current_dir(current_dir());

        // Set up input and output file.
        let mut texture_asset_path = config_path.with_file_name("");
        texture_asset_path.push(self.path);
        rerun_if_changed(&texture_asset_path);

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
                let bump_texture = image::open(&texture_asset_path)
                    .expect("Failed to open bump texture")
                    .into_rgb8();

                // File paths.
                let cached_normal_path = crate::convert_content_to_output_dir(&texture_asset_path, OutputDir::Cache)
                    .unwrap()
                    .with_extension("normal.png");
                let cached_hash_path = cached_normal_path.with_extension("hash");
                crate::create_required_folders(&cached_normal_path);

                // If the normal map already exists, check if it's up to date.
                let mut hasher = sha3::Sha3_256::new();
                hasher.update(bump_texture.as_raw());
                let hash = hasher.finalize();
                if std::fs::exists(&cached_normal_path).unwrap() {
                    // Check the hash of the bump texture to ensure the normal map is only rebuilt when the bump texture changes.
                    if let Ok(hash_file) = std::fs::read_to_string(&cached_hash_path) {
                        if hash_file == format!("{:x}", hash) {
                            // Normal map is up to date.
                            return;
                        }
                    }
                }
                // Write out the hash.
                std::fs::write(&cached_hash_path, format!("{:x}", hash)).expect("Failed to write new hash to disk");

                let normal_texture =
                    normal_heights::map_normals_with_strength(&DynamicImage::ImageRgb8(bump_texture), 1.0);

                // Save the normal texture to disk.
                normal_texture
                    .save(&cached_normal_path)
                    .expect("Failed to save normal texture to disk");

                texture_asset_path = cached_normal_path;
                command
                    .args(["--assign-oetf", "linear"])
                    .args(["--format", "R8G8B8_UNORM"])
                    .arg("--normal-mode");
            }
            TextureType::Normal => {
                command.args(["--normal-mode", "--format", "R8G8B8_UNORM"]);
            }
        };

        // Set up UASTC compression.
        command.args(["--encode", "uastc", "--uastc-rdo"]);

        if debug {
            command.args(["--uastc-quality", "2"]).args(["--zstd", "10"]);
        } else {
            command.args(["--uastc-quality", "5"]).args(["--zstd", "18"]);
        }

        if self.mipmap {
            command.arg("--generate-mipmap");
        } else {
            unimplemented!("Mipmaps must be generated for now.");
        }

        // Create output paths.
        let mut output_file_path = crate::convert_content_to_output_dir(config_path, OutputDir::Build).unwrap();
        output_file_path.set_file_name(filename);
        output_file_path.set_extension("ktx2");
        crate::create_required_folders(&output_file_path);

        command.arg(texture_asset_path).arg(output_file_path);

        crate::handle_command_result(command.output(), "Texture build failed", "ktx");
    }
}

pub fn build_textures(glob_texture_paths: &[&str], debug: bool) {
    for glob_path in glob_texture_paths {
        for path in glob(&crate::join(CONTENT_DIR, glob_path)).expect("Failed to read texture glob pattern") {
            let config_path = path.unwrap();
            rerun_if_changed(&config_path);

            let texture_config_source =
                std::fs::read_to_string(&config_path).expect("Failed to read texture config file.");
            let configs: HashMap<&str, Texture> =
                load_config(&texture_config_source).expect("Failed to load texture config.");
            for (filename, config) in configs {
                config.build(&config_path, filename, debug);
            }
        }
    }
}
