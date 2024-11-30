// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use const_format::concatcp;
use glob::glob;
use itertools::{Either, Itertools};

use crate::engine::GalaxyEngine;
use crate::pipelines;
use crate::pipelines::config::{
    FragmentShaderModuleCache, PipelineConfig, PipelineLayoutCache, VertexShaderModuleCache,
};
use crate::pipelines::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline};
use crate::pipelines::PipelineLayout;
use crate::utils::{ArcFinalOwner, EntryExt};
use crate::vulkan::descriptors::DescriptorSetLayout;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::shader::ShaderModule;

#[derive(thiserror::Error, Debug)]
pub enum PipelineManagerError {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
}

// TODO: Pipeline cache: https://zeux.io/2019/07/17/serializing-pipeline-cache/.
// Could use an arena allocator for all the Arcs created here.
pub struct PipelineManager {
    loader: SharedDeviceLoader,
    pub(crate) scene_descriptor_set_layout: DescriptorSetLayout,
    graphics_pipelines: HashMap<String, ArcFinalOwner<GraphicsPipeline>>,
    compute_pipelines: HashMap<String, ArcFinalOwner<ComputePipeline>>,
}

impl PipelineManager {}

impl PipelineManager {
    const PIPELINE_CONFIG_GLOB: &'static str = concatcp!(GalaxyEngine::SHADER_PATH, "**/*.pipeline.ron");

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
                .descriptor_count(GalaxyEngine::NUM_TEXTURES as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Material data storage:
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        ]
    }

    pub fn new(device: &Device, msaa_samples: vk::SampleCountFlags) -> Result<Self, PipelineManagerError> {
        // Create level descriptor set layout.
        // TODO: Pipeline manager handles descriptor set layout lifetimes?
        let scene_descriptor_set_layout =
            DescriptorSetLayout::new(&device, &Self::scene_descriptor_set_layout_bindings())?;

        // Find and load pipeline configs.
        let (graphics_configs, _compute_configs): (Vec<_>, Vec<_>) = glob(Self::PIPELINE_CONFIG_GLOB)
            .expect("Failed to read pipeline glob pattern")
            .filter_map(|path| {
                let name = (match path.as_ref() {
                    Ok(path) => path,
                    Err(err) => err.path(),
                })
                .file_name()?
                .to_owned();

                // Nested function for error-handling.
                fn load_config(path: glob::GlobResult) -> Result<PipelineConfig, PipelineManagerError> {
                    let config_str = match path {
                        Ok(path) => std::fs::read_to_string(&path),
                        Err(err) => Err(err.into_error()),
                    }?;
                    let config = pipelines::config::load_config(&config_str)?;
                    Ok(config)
                }

                match load_config(path) {
                    Ok(config) => Some(config),
                    Err(err) => {
                        log::error!("Failed to load pipeline config for pipeline {name:?} ({err})");
                        None
                    }
                }
            })
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
            let bindings = config.layout.bindings();
            let push_constant_range = bindings.push_constant.as_ref().map(|c| c.push_constant_range());
            pipeline_layouts
                .entry(bindings.push_constant)
                .try_or_insert_with::<vk::Result, _>(|| {
                    let layout = PipelineLayout::new(
                        &device,
                        Some(&[scene_descriptor_set_layout.handle()]),
                        push_constant_range.as_ref(),
                    )?;
                    Ok(Arc::new(layout))
                })?;

            // Load shaders.
            vertex_shaders
                .entry(config.shaders.vertex.id.clone())
                .try_or_insert_with(|| ShaderModule::new(&device, &config.shaders.vertex.id))?;
            fragment_shaders
                .entry(config.shaders.fragment.id.clone())
                .try_or_insert_with(|| ShaderModule::new(&device, &config.shaders.fragment.id))?;
        }

        let graphics_pipelines = GraphicsPipeline::batch_new(
            &device,
            &pipeline_layouts,
            &vertex_shaders,
            &fragment_shaders,
            &graphics_configs,
            msaa_samples,
        )?
        .into_iter()
        .zip(&graphics_configs)
        .map(|(pipeline, config)| (config.name.clone(), ArcFinalOwner::new(pipeline)))
        .collect();

        Ok(Self {
            loader: device.cloned_loader(),
            scene_descriptor_set_layout,
            graphics_pipelines,
            compute_pipelines: HashMap::new(),
        })
    }

    pub fn get_graphics_pipeline(&self, name: &String) -> Option<Arc<GraphicsPipeline>> {
        self.graphics_pipelines.get(name).as_deref().map(ArcFinalOwner::clone)
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        fn destroy_pipeline<T: Pipeline>(loader: &ash::Device, id: String, pipeline: &mut ArcFinalOwner<T>) {
            unsafe { pipeline.destroy_as_final(|pipeline| loader.destroy_pipeline(pipeline.handle(), None)) }
                .unwrap_or_else(|_| log::error!("Pipeline manager not final owner of pipeline: {id}."));
        }
        for (id, mut pipeline) in self.graphics_pipelines.drain() {
            destroy_pipeline(&self.loader, id, &mut pipeline);
        }
        for (id, mut pipeline) in self.compute_pipelines.drain() {
            destroy_pipeline(&self.loader, id, &mut pipeline);
        }
    }
}
