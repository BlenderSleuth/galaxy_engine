use glob::glob;
use std::path::Path;
use std::process::Command;

#[macro_export]
macro_rules! print_warning {
    ($($tokens: tt)*) => {
        println!("cargo::warning={}", format!($($tokens)*))
    }
}

fn compile_stage(path: &Path, shader_model: &str, entry: &str, output_ext: &str, debug: bool) {
    let current_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_path = path.with_extension(format!("{output_ext}.spv").as_str());
    println!("cargo::rerun-if-changed={output_path:?}");
    let mut command = Command::new("dxc");
    command
        .current_dir(current_dir)
        .arg("-spirv")
        .arg(format!("-T {shader_model}").as_str())
        .arg(format!("-E {entry}"))
        .arg(path)
        .arg(format!("-Fo {}", output_path.to_str().unwrap()).as_str());
    if debug {
        command.arg("-Zi");
        command.arg("-fspv-debug=vulkan-with-source");
    }

    match command.output() {
        Ok(output) => {
            if output.stderr.len() > 0 {
                let stderr = String::from_utf8_lossy(&output.stderr);
                for line in stderr.lines() {
                    print_warning!("{line}");
                }
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
            panic!("Shader compile failed. Make sure you have dxc installed.");
        }
    };
}

struct ShaderStages {
    vertex: bool,
    fragment: bool,
    compute: bool,
}

impl ShaderStages {
    fn from_source(source: &str) -> Self {
        let mut result = Self {
            vertex: false,
            fragment: false,
            compute: false,
        };
        if source.contains("mainVS") {
            result.vertex = true;
        }
        if source.contains("mainFS") {
            result.fragment = true;
        }
        if source.contains("mainCS") {
            result.compute = true;
        }
        result
    }

    fn compile(&self, path: &Path, debug: bool) {
        println!("cargo::rerun-if-changed={path:?}");
        if self.vertex {
            compile_stage(path, "vs_6_0", "mainVS", "vert", debug);
        }
        if self.fragment {
            compile_stage(path, "ps_6_0", "mainFS", "frag", debug);
        }
        if self.compute {
            compile_stage(path, "cs_6_0", "mainCS", "comp", debug);
        }
    }
}

// Compile all shaders in the given glob pattern. Relies on dxc being installed.
// Relative to CARGO_MANIFEST_DIR.
pub fn compile_shaders(glob_shader_paths: &[&str], debug: bool) {
    //let glob_shader_paths: Vec<String> = glob_shader_paths.iter().map(|s| format!("{current_dir}/{s}")).collect();
    for glob_path in glob_shader_paths {
        for path in glob(glob_path).expect("Failed to read shader glob pattern") {
            let shader_path = path.unwrap();
            let shader_source = std::fs::read_to_string(&shader_path).unwrap();
            ShaderStages::from_source(&shader_source).compile(&shader_path, debug);
        }
    }
}