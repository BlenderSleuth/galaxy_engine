// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;
use std::process::Command;

use glob::glob;

#[macro_export]
macro_rules! print_warning {
    ($($tokens: tt)*) => {
        println!("cargo::warning={}", format!($($tokens)*))
    }
}

#[derive(Copy, Clone)]
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

    fn compile(&self, input_file_path: &Path, debug: bool) {
        if !self.any_stages() {
            return;
        }

        let current_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!(
            "cargo::rerun-if-changed={}",
            current_dir.clone() + "/" + input_file_path.to_str().unwrap()
        );

        // Stage / output path pairs. If adding other stages, make a macro.
        let output_file_path = input_file_path.with_extension("spv");
        //let vert_output_path = input_file_path.with_extension("vert.spv");
        //let frag_output_path = input_file_path.with_extension("frag.spv");
        //let comp_output_path = input_file_path.with_extension("comp.spv");

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
            .current_dir(current_dir)
            .arg(input_file_path)
            .args(["-target", "spirv"])
            .args(["-profile", shader_model])
            .arg("-fvk-use-entrypoint-name")
            .args(file_stage_args)
            .args(["-I", "content/shaders"])
            .args(["-o", output_file_path.to_str().unwrap()]);

        if debug {
            command.arg("-g2").arg("-O0");
        } else {
            command.arg("-g0").arg("-O2");
        }

        match command.output() {
            Ok(output) => {
                if output.stderr.len() > 0 {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    for line in stderr.lines() {
                        print_warning!("{line}");
                    }
                    panic!("Shader compile failed.");
                }
                if output.stdout.len() > 0 {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        print_warning!("{line}");
                    }
                }
            }
            Err(err) => {
                print_warning!("{err}");
                panic!("Shader compile failed. Make sure you have slangc installed.");
            }
        };
    }
}

// Compile all shaders in the given glob pattern. Relies on dxc being installed.
// Paths are relative to CARGO_MANIFEST_DIR.
pub fn compile_shaders(glob_shader_paths: &[&str], debug: bool) {
    for glob_path in glob_shader_paths {
        for path in glob(glob_path).expect("Failed to read shader glob pattern") {
            let shader_path = path.unwrap();
            let shader_source = std::fs::read_to_string(&shader_path).unwrap();
            ShaderStages::from_source(&shader_source).compile(&shader_path, debug);
        }
    }
}
