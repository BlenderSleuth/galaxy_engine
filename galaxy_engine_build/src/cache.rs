// Copyright (c) 2024-2025 Ben Sutherland.

use std::io::Read;
use std::path::Path;

use sha3::Digest;

use crate::OutputDir;

pub fn bincode_hash<E: bincode::Encode, D: Digest>(data: &E, hasher: &mut D) {
    let bytes = bincode::encode_to_vec(data, bincode::config::standard()).expect("Failed to encode config");
    hasher.update(&bytes);
}

/// Checks if the built asset already exists in the cache.
/// Will write out a hash file if the asset is not up to date.
pub fn exists_in_cache<E: bincode::Encode>(
    config: &E,
    source_asset_path: &Path,
    built_filename: &str,
    source_asset_bytes: Option<&[u8]>,
) -> bool {
    let cache_asset_file_path =
        crate::convert_content_to_output_dir(source_asset_path, built_filename, OutputDir::Cache).unwrap();
    let cache_hash_file_path = cache_asset_file_path.with_extension("hash");

    // Load the source asset so we can hash it.
    let mut asset_data = Vec::new();
    let asset_data_slice = if let Some(source_asset_bytes) = source_asset_bytes {
        source_asset_bytes
    } else {
        if let Ok(mut file) = std::fs::File::open(source_asset_path) {
            file.read_to_end(&mut asset_data).expect("Failed to read source asset file");
        } else {
            return false;
        };
        //_asset_data = std::fs::read(source_asset_path).expect("Failed to read source asset file");
        &asset_data
    };

    let hash = {
        let mut hasher = sha3::Sha3_256::new();
        // Hash both the source asset and the config.
        hasher.update(asset_data_slice);
        bincode_hash(config, &mut hasher);
        hasher.finalize()
    };
    let hash_str = format!("{hash:x}");

    if std::fs::exists(&cache_asset_file_path).unwrap_or(false) {
        if let Ok(hash_str_from_cache) = std::fs::read_to_string(&cache_hash_file_path) {
            if hash_str_from_cache == hash_str {
                // Texture is up to date.
                return true;
            }
        }
    }

    // Write out the hash.
    std::fs::create_dir_all(cache_asset_file_path.parent().unwrap()).expect("Failed to create cache folder");
    std::fs::write(&cache_hash_file_path, hash_str).expect("Failed to write new hash to disk");

    false
}

fn _files_are_identical(file_path_a: &Path, file_path_b: &Path) -> bool {
    // Open files:
    let Ok(file_a) = std::fs::File::open(file_path_a) else {
        return false;
    };
    let Ok(file_b) = std::fs::File::open(file_path_b) else {
        return false;
    };

    // Check size first:
    let metadata_a = file_a.metadata().expect("Failed to get metadata for file A");
    let metadata_b = file_b.metadata().expect("Failed to get metadata for file B");
    if metadata_a.len() != metadata_b.len() {
        return false;
    }

    // Stream chunks and compare:
    let mut reader_a = std::io::BufReader::new(file_a);
    let mut reader_b = std::io::BufReader::new(file_b);
    let mut buffer_a = [0; 1024];
    let mut buffer_b = [0; 1024];
    loop {
        let Ok(bytes_read_a) = reader_a.read(&mut buffer_a) else {
            return false;
        };
        let Ok(bytes_read_b) = reader_b.read(&mut buffer_b) else {
            return false;
        };
        if bytes_read_a != bytes_read_b {
            return false;
        }
        if bytes_read_a == 0 {
            // Reached end of file.
            return true;
        }
        if buffer_a[..bytes_read_a] != buffer_b[..bytes_read_b] {
            return false;
        }
    }
}

pub fn copy_from_cache_to_build(output_path: &Path, built_filename: &str) {
    let cache_file_path = crate::convert_content_to_output_dir(output_path, built_filename, OutputDir::Cache).unwrap();
    let build_file_path = crate::convert_content_to_output_dir(output_path, built_filename, OutputDir::Build).unwrap();

    // This check seemed slower than just copying the file in every case.
    //if files_are_identical(&cache_file_path, &build_file_path) {
    //    // The asset is already up to date in the build folder.
    //    return;
    //}

    // Copy the asset from the cache to the build folder.
    std::fs::create_dir_all(build_file_path.parent().unwrap()).expect("Failed to create build folder for asset");
    std::fs::copy(cache_file_path, build_file_path)
        .unwrap_or_else(|_| panic!("Failed to copy {output_path:?} from cache to build"));
}
