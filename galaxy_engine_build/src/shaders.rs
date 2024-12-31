// Copyright (c) 2024-2025 Ben Sutherland.

use std::path::Path;
use std::process::Command;

use glob::glob;

use crate::{convert_content_to_output_dir, current_dir, rerun_if_changed, OutputDir, CONTENT_DIR};

struct ShaderStages {
    vertex: bool,
    fragment: bool,
    compute: bool,
}

impl ShaderStages {
    fn from_source(source: &str) -> Self {
        Self {
            vertex: source.contains("shader(\"vertex\")"),
            fragment: source.contains("shader(\"fragment\")"),
            compute: source.contains("shader(\"compute\")"),
        }
    }

    fn any_stages(&self) -> bool {
        self.vertex || self.fragment || self.compute
    }

    fn compile(&self, input_file_path: &Path) {
        // Rerun if any shader file is changed, even if it's just a module.
        rerun_if_changed(input_file_path);

        if !self.any_stages() {
            return;
        }

        // Stage / output path pairs. If adding other stages, make a macro.
        let output_filename = input_file_path
            .with_extension("spv")
            .file_name()
            .unwrap()
            .to_os_string()
            .into_string()
            .unwrap();
        let output_file_path =
            convert_content_to_output_dir(input_file_path, &output_filename, OutputDir::Build).unwrap();
        //output_file_path.set_extension("spv");

        crate::create_required_folders(&output_file_path);
        //let vert_output_path = input_file_path.with_extension("vert.spv");
        //let frag_output_path = input_file_path.with_extension("frag.spv");
        //let comp_output_path = input_file_path.with_extension("comp.spv");

        //let global_session = slang::GlobalSession::new().unwrap();
        //let options = slang::OptionsBuilder::new()
        //    .optimization(slang::OptimizationLevel::Maximal)
        //    .matrix_layout_row(true);

        let mut file_stage_args = Vec::new();
        if self.vertex {
            file_stage_args.extend(["-entry", "mainVS", "-stage", "vertex"]);
            //let vert_output_path = vert_output_path.to_str().unwrap();
            //file_stage_args.extend(["-o", vert_output_path]);
            //println!("cargo::rerun-if-changed={vert_output_path}");
        }
        if self.fragment {
            file_stage_args.extend(["-entry", "mainFS", "-stage", "fragment"]);
            //let frag_output_path = frag_output_path.to_str().unwrap();
            //file_stage_args.extend(["-o", frag_output_path]);
            //println!("cargo::rerun-if-changed={frag_output_path}");
        }
        if self.compute {
            file_stage_args.extend(["-entry", "mainCS", "-stage", "compute"]);
            //let comp_output_path = comp_output_path.to_str().unwrap();
            //file_stage_args.extend(["-o", comp_output_path]);
            //println!("cargo::rerun-if-changed={comp_output_path}");
        }
        //print_warning!("{file_stage_args:?}");

        let shader_model = "spirv_1_5";

        let mut command = Command::new("slangc");
        command
            .current_dir(current_dir())
            .arg(input_file_path)
            .args(["-target", "spirv"])
            .args(["-profile", shader_model])
            .arg("-fvk-use-entrypoint-name")
            .arg("-matrix-layout-row-major")
            .args(file_stage_args)
            .args(["-I", "content/shaders"])
            .args(["-o", output_file_path.to_str().unwrap()]);

        if cfg!(feature = "packaged") {
            command.arg("-g0").arg("-O3");
        } else {
            command
                .arg("-g2")
                .arg("-O0")
                .args(["-capability", "SPV_KHR_non_semantic_info"]);
        }

        crate::handle_command_result(command.output(), "Shader compile failed", "slangc");
    }
}

// Compile all shaders in the given glob pattern. Relies on slangc being installed.
// Paths are relative to CARGO_MANIFEST_DIR.
pub fn compile_shaders(glob_shader_paths: &[&str]) {
    for glob_path in glob_shader_paths {
        for path in glob(&crate::str_path_join(CONTENT_DIR, glob_path)).expect("Failed to read shader glob pattern.") {
            let shader_path = path.unwrap();
            let shader_source = std::fs::read_to_string(&shader_path).unwrap();
            ShaderStages::from_source(&shader_source).compile(&shader_path);
        }
    }
}
