// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;

use ash::vk;
use basis_universal::{DecodeFlags, SliceParametersUastc, TranscodeError, TranscoderBlockFormat};
use ktx2::{BasicDataFormatDescriptor, DataFormatDescriptorHeader};

use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemoryError;
use crate::vulkan::image::{Image, ImageDimensions};

#[derive(Debug, thiserror::Error)]
pub enum TextureError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("KTX2 parse error: {0}")]
    Ktx2Error(#[from] ktx2::ParseError),
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("Unsupported texture format: {0}")]
    UnsupportedTextureFormat(String),
    #[error("Transcode error: {0:?}")]
    TranscodeError(TranscodeError),
}

mod ktx2 {
    pub use ktx2::*;

    //pub const DF_CHANNEL_UASTC_RGB: u8 = 0;
    pub const DF_CHANNEL_UASTC_RGBA: u8 = 3;
    //pub const DF_CHANNEL_UASTC_RRR: u8 = 4;
    pub const DF_CHANNEL_UASTC_RRRG: u8 = 5;
    //pub const DF_CHANNEL_UASTC_RG: u8 = 6;
}

pub struct Texture {
    image: Image,
}

impl Texture {
    pub fn new_from_ktx2_file(
        name: &str,
        path: &Path,
        device: &Device,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, TextureError> {
        // Load texture.
        let texture_data = std::fs::read(path)?;
        let image = ktx2::Reader::new(texture_data)?;
        let header = image.header();

        // Check file is encoded as zstd compressed UASTC.
        if header.supercompression_scheme != Some(ktx2::SupercompressionScheme::Zstandard) {
            return Err(TextureError::UnsupportedTextureFormat(format!(
                "Unsupported super-compression scheme: {:?}",
                header.supercompression_scheme
            )));
        }

        let mut found_dfd = false;
        let mut has_alpha = false;
        let mut is_normal_map = false;
        for dfd in image.data_format_descriptors() {
            if dfd.header == DataFormatDescriptorHeader::BASIC {
                let basic = BasicDataFormatDescriptor::parse(dfd.data)?;
                if basic.header.color_model != Some(ktx2::ColorModel::UASTC) {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "Unsupported color model: {:?} for texture: {name}",
                        basic.header.color_model
                    )));
                }
                if let Some(sample) = basic.sample_information().next() {
                    has_alpha = sample.channel_type == ktx2::DF_CHANNEL_UASTC_RGBA;
                    is_normal_map = sample.channel_type == ktx2::DF_CHANNEL_UASTC_RRRG;
                } else {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "No sample information found for texture: {name}"
                    )));
                }
                found_dfd = true;
            }
        }
        if !found_dfd {
            return Err(TextureError::UnsupportedTextureFormat(format!(
                "No basic data format descriptor found for texture: {name}"
            )));
        }

        // Uncompress mip levels.
        let mut mip_ranges = Vec::with_capacity(image.levels().len());
        let mut decoded_data = Vec::with_capacity(image.levels().map(|l| l.uncompressed_byte_length as usize).sum());
        let old_capacity = decoded_data.capacity();
        for level in image.levels() {
            let start = decoded_data.len();
            zstd::stream::copy_decode(&mut std::io::Cursor::new(level.data), &mut decoded_data)?;
            let range = start..decoded_data.len();
            assert_eq!(range.len(), level.uncompressed_byte_length as usize);
            mip_ranges.push(range);
        }
        debug_assert_eq!(decoded_data.capacity(), old_capacity);

        let transcoder = basis_universal::LowLevelUastcTranscoder::new();
        let mip_level_data = mip_ranges
            .into_iter()
            .enumerate()
            .map(|(i, range)| {
                let level_width = header.pixel_width >> i;
                let level_height = header.pixel_height >> i;

                // Ignore zero size mip levels.
                if level_width == 0 || level_height == 0 {
                    return Ok(Vec::new());
                }

                let num_blocks_x = (level_width + 3) / 4;
                let num_blocks_y = (level_height + 3) / 4;

                let params = SliceParametersUastc {
                    num_blocks_x,
                    num_blocks_y,
                    has_alpha,
                    original_width: level_width,
                    original_height: level_height,
                };

                // Transcode mip levels.
                transcoder.transcode_slice(
                    &decoded_data[range],
                    params,
                    DecodeFlags::HIGH_QUALITY,
                    if is_normal_map {
                        TranscoderBlockFormat::BC5
                    } else {
                        TranscoderBlockFormat::BC7
                    },
                )
            })
            .collect::<Result<Vec<_>, TranscodeError>>()
            .map_err(TextureError::TranscodeError)?;
        let mip_levels = mip_level_data
            .iter()
            .filter_map(|data| {
                let data = data.as_slice();
                if data.is_empty() {
                    None
                } else {
                    Some(data)
                }
            })
            .collect::<Vec<_>>();

        let extent = vk::Extent2D {
            width: header.pixel_width,
            height: header.pixel_height,
        };

        let texture_image = Image::new_from_mip_levels(
            debug::debug_only_name!("{name} texture"),
            device,
            cmd_pool,
            &mip_levels,
            ImageDimensions::Type2D(extent),
            if is_normal_map {
                vk::Format::BC5_UNORM_BLOCK
            } else {
                vk::Format::BC7_SRGB_BLOCK
            },
        )?;

        Ok(Self { image: texture_image })
    }

    pub fn image(&self) -> &Image {
        &self.image
    }
}
