// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use galaxy_engine_utils::config::load_config;
use glob::glob;
use serde::Deserialize;

use crate::{current_dir, rerun_if_changed, CONTENT_DIR};

#[derive(Deserialize, Debug, Copy, Clone)]
enum TextureComponents {
    R,
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
        // Run ktx create. https://github.khronos.org/KTX-Software/ktxtools/ktx_create.html.
        let mut command = Command::new("ktx");
        command.arg("create").current_dir(current_dir());

        // Set up UASTC compression.
        command.args(["--encode", "uastc", "--uastc-rdo"]);

        if debug {
            command.args(["--uastc-quality", "2"]).args(["--zstd", "10"]);
        } else {
            command.args(["--uastc-quality", "5"]).args(["--zstd", "22"]);
        }

        match self.ty {
            TextureType::Colour(dimensions) => {
                command
                    .args(["--assign-oetf", "sRGB"])
                    .args(["--assign-primaries", "sRGB"])
                    .args([
                        "--format",
                        match dimensions {
                            TextureComponents::R => "R8_SRGB",
                            TextureComponents::Rg => "R8G8_SRGB",
                            TextureComponents::Rgb => "R8G8B8_SRGB",
                            TextureComponents::Rgba => "R8G8B8A8_SRGB",
                        },
                    ]);
            }
            TextureType::Linear(dimensions) => {
                command.args([
                    "--format",
                    match dimensions {
                        TextureComponents::R => "R8_UNORM",
                        TextureComponents::Rg => "R8G8_UNORM",
                        TextureComponents::Rgb => "R8G8B8_UNORM",
                        TextureComponents::Rgba => "R8G8B8A8_UNORM",
                    },
                ]);
            }
            TextureType::Normal => {
                command.args(["--normal-mode", "--normalize"]);
            }
        };

        if self.mipmap {
            command.arg("--generate-mipmap");
        } else {
            unimplemented!("Mipmaps must be generated for now.");
        }

        // Setup input and output file.
        let mut texture_asset_path = config_path.with_file_name("");
        texture_asset_path.push(self.path);
        rerun_if_changed(&texture_asset_path);

        // Create output paths.
        let mut output_file_path = crate::convert_content_to_build_dir(config_path).unwrap();
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
