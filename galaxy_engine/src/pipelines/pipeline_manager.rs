// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use const_format::concatcp;
use glob::glob;
use itertools::{Either, Itertools};

use crate::engine::GalaxyEngine;
use crate::pipelines::config::{PipelineConfig, PushConstantBinding};
use crate::pipelines::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline};
use crate::pipelines::PipelineLayout;
use crate::textures::TextureManager;
use crate::utils::{ArcFinalOwner, EntryExt};
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::shader::{FragmentShaderStage, ShaderModule, VertexShaderStage};

#[derive(thiserror::Error, Debug)]
pub enum PipelineManagerError {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
}

pub type PipelineLayoutCache = HashMap<Option<PushConstantBinding>, Arc<PipelineLayout>>;
pub type VertexShaderModuleCache<'a> = HashMap<&'a str, ShaderModule<VertexShaderStage>>;
pub type FragmentShaderModuleCache<'a> = HashMap<&'a str, ShaderModule<FragmentShaderStage>>;

// TODO: Pipeline cache: https://zeux.io/2019/07/17/serializing-pipeline-cache/.
// Could use an arena allocator for all the Arcs created here.
pub struct PipelineManager {
    loader: SharedDeviceLoader,
    pub(crate) scene_set_layout: vk::DescriptorSetLayout,
    pipeline_layouts: PipelineLayoutCache,
    graphics_pipelines: HashMap<Arc<str>, ArcFinalOwner<GraphicsPipeline>>,
    compute_pipelines: HashMap<Arc<str>, ArcFinalOwner<ComputePipeline>>,
}

impl PipelineManager {
    pub const SHADER_DIR: &'static str = "shaders/";
    pub const SHADER_PATH: &'static str = concatcp!(GalaxyEngine::CONTENT_PATH, PipelineManager::SHADER_DIR);
    pub const BUILT_SHADER_PATH: &'static str = concatcp!(GalaxyEngine::BUILT_PATH, PipelineManager::SHADER_DIR);
    const PIPELINE_CONFIG_GLOB: &'static str = "**/*.pipeline.ron";
    pub const MAX_PIPELINES_PER_LAYOUT: usize = 512;

    pub fn create_descriptor_set_layout(
        device: &Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> VkResult<vk::DescriptorSetLayout> {
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        unsafe { device.loader().create_descriptor_set_layout(&info, None) }
    }

    fn scene_descriptor_set_layout_bindings() -> [vk::DescriptorSetLayoutBinding<'static>; 4] {
        [
            // Scene uniforms:
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                ),
            // Transforms:
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            // Array of textures:
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(TextureManager::MAX_TEXTURES as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Material data storage:
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_count(Self::MAX_PIPELINES_PER_LAYOUT as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ]
    }

    //fn material_data_descriptor_set_layout_bindings() -> [vk::DescriptorSetLayoutBinding<'static>; 1] {
    //    [
    //        // Material data storage:
    //        vk::DescriptorSetLayoutBinding::default()
    //            .binding(0)
    //            .descriptor_count(1)
    //            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
    //            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    //    ]
    //}

    pub fn new(device: &Device, msaa_samples: vk::SampleCountFlags) -> Result<Self, PipelineManagerError> {
        // Create level descriptor set layout.
        let scene_set_layout =
            Self::create_descriptor_set_layout(device, &Self::scene_descriptor_set_layout_bindings())?;
        //let material_set_layout =
        //    Self::create_descriptor_set_layout(device, &Self::material_data_descriptor_set_layout_bindings())?;

        // Find and load pipeline configs. TODO: Add support for game-specific pipeline configs.
        let config_strings = glob(
            Path::new(Self::SHADER_PATH)
                .join(Self::PIPELINE_CONFIG_GLOB)
                .to_str()
                .unwrap(),
        )
        .expect("Failed to read pipeline glob pattern")
        .filter_map(|path| {
            let name = (match path.as_ref() {
                Ok(path) => path,
                Err(err) => err.path(),
            })
            .file_name()?
            .to_str()?
            .to_owned();

            // Nested function for error-handling.
            fn load_config(path: glob::GlobResult) -> Result<String, PipelineManagerError> {
                Ok(std::fs::read_to_string(&path.map_err(|e| e.into_error())?)?)
            }

            let config_str = match load_config(path) {
                Ok(config) => Some(config),
                Err(err) => {
                    log::error!("Failed to read pipeline config for pipeline {name} ({err}).");
                    None
                }
            }?;

            Some((name, config_str))
        })
        .collect::<Vec<_>>();

        let (graphics_configs, _compute_configs): (Vec<_>, Vec<_>) = config_strings
            .iter()
            .filter_map(
                |(name, config_str)| match crate::utils::load_config::<PipelineConfig>(config_str) {
                    Ok(config) => Some(config),
                    Err(err) => {
                        log::error!("Failed to parse pipeline config for pipeline {name} ({err})");
                        None
                    }
                },
            )
            .partition_map(|config| match config {
                PipelineConfig::Graphics(graphics) => Either::Left(graphics),
                PipelineConfig::Compute(compute) => Either::Right(compute),
            });

        // Compile graphics pipelines. For lots of graphics pipelines, could use a graphics pipeline library for speedup:
        // https://www.khronos.org/blog/reducing-draw-time-hitching-with-vk-ext-graphics-pipeline-library.
        let mut pipeline_layouts = PipelineLayoutCache::new();
        let mut vertex_shaders = VertexShaderModuleCache::new();
        let mut fragment_shaders = FragmentShaderModuleCache::new();

        for config in &graphics_configs {
            // Find or construct pipeline layout.
            let push_constant = config.layout.push_constant;
            let push_constant_range = push_constant.as_ref().map(|c| c.push_constant_range());
            pipeline_layouts
                .entry(push_constant)
                .try_or_insert_with::<vk::Result, _>(|| {
                    let layout = PipelineLayout::new(device, Some(&[scene_set_layout]), push_constant_range.as_ref())?;
                    Ok(Arc::new(layout))
                })?;

            // Load shaders.
            vertex_shaders
                .entry(config.shaders.vertex.id)
                .try_or_insert_with(|| ShaderModule::new(device, config.shaders.vertex.id))?;
            fragment_shaders
                .entry(config.shaders.fragment.id)
                .try_or_insert_with(|| ShaderModule::new(device, config.shaders.fragment.id))?;
        }

        log::info!("Compiling graphics pipelines...");
        let compilation_start = std::time::Instant::now();
        let graphics_pipelines = GraphicsPipeline::batch_new(
            device,
            &pipeline_layouts,
            vertex_shaders,
            fragment_shaders,
            graphics_configs,
            msaa_samples,
        )?;
        log::info!("Compiled graphics pipelines in {:?}", compilation_start.elapsed());

        // Collect to hashmap.
        let graphics_pipelines = graphics_pipelines
            .into_iter()
            .map(|pipeline| (pipeline.cloned_name(), ArcFinalOwner::new(pipeline)))
            .collect();

        Ok(Self {
            loader: device.cloned_loader(),
            scene_set_layout,
            pipeline_layouts,
            graphics_pipelines,
            compute_pipelines: HashMap::new(),
        })
    }

    pub fn get_graphics_pipeline(&self, name: &str) -> Option<&GraphicsPipeline> {
        self.graphics_pipelines.get(name).map(ArcFinalOwner::as_ref)
    }

    pub fn get_cloned_graphics_pipeline(&self, name: &str) -> Option<Arc<GraphicsPipeline>> {
        self.graphics_pipelines.get(name).map(ArcFinalOwner::clone)
    }

    pub fn iter_graphics_pipelines(&self) -> impl Iterator<Item = &GraphicsPipeline> {
        self.graphics_pipelines.values().map(ArcFinalOwner::as_ref)
    }

    //pub fn num_layouts(&self) -> usize {
    //    self.pipeline_layouts.len()
    //}

    pub fn get_layout(&self, binding: Option<PushConstantBinding>) -> Option<&PipelineLayout> {
        self.pipeline_layouts.get(&binding).map(Arc::as_ref)
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        // Destroy pipelines.
        fn destroy_pipeline<T: Pipeline>(loader: &ash::Device, id: &str, pipeline: &mut ArcFinalOwner<T>) {
            unsafe { pipeline.destroy_as_final(|pipeline| loader.destroy_pipeline(pipeline.handle(), None)) }
                .unwrap_or_else(|_| log::error!("Pipeline manager not final owner of pipeline: {id}."));
        }
        for (id, mut pipeline) in self.graphics_pipelines.drain() {
            destroy_pipeline(&self.loader, &id, &mut pipeline);
        }
        for (id, mut pipeline) in self.compute_pipelines.drain() {
            destroy_pipeline(&self.loader, &id, &mut pipeline);
        }

        // Destroy descriptor set layout.
        unsafe { self.loader.destroy_descriptor_set_layout(self.scene_set_layout, None) }
        //unsafe {
        //    self.loader
        //        .destroy_descriptor_set_layout(self.material_set_layout, None)
        //}
    }
}
