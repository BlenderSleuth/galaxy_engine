// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use const_format::concatcp;
use glob::glob;
use itertools::{Either, Itertools};
use path_slash::PathExt;

use crate::engine::GalaxyEngine;
use crate::pipelines::config::{ComputeResourceType, PipelineConfig, PushConstantBinding};
use crate::pipelines::pipeline::{
    ComputePipeline, ComputePipelineCreateResources, GraphicsPipeline, GraphicsPipelineCreateResources, Pipeline,
};
use crate::textures::TextureManager;
use crate::utils::{ArcFinalOwner, EntryExt};
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::shader::{shader_stage, ShaderModule};

fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> VkResult<vk::DescriptorSetLayout> {
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
    unsafe { device.loader.create_descriptor_set_layout(&info, None) }
}

fn get_or_create_compute_descriptor_set_layout(
    device: &Device,
    cache: &mut ComputeDescriptorSetLayoutCache,
    bindings: Vec<ComputeResourceType>,
) -> VkResult<vk::DescriptorSetLayout> {
    match cache.entry(bindings) {
        Entry::Occupied(entry) => Ok(*entry.get()),
        Entry::Vacant(entry) => {
            let bindings = entry
                .key()
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(ty.descriptor_type())
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();

            let layout = create_descriptor_set_layout(device, &bindings)?;
            entry.insert(layout);
            Ok(layout)
        }
    }
}

fn create_pipeline_layout(
    device: &Device,
    push_constant_range: Option<&[vk::PushConstantRange]>,
    descriptor_set_layout: Option<&[vk::DescriptorSetLayout]>,
) -> VkResult<vk::PipelineLayout> {
    let mut pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
    if let Some(descriptor_set_layout) = descriptor_set_layout {
        pipeline_layout_info = pipeline_layout_info.set_layouts(descriptor_set_layout);
    }
    if let Some(push_constant_range) = push_constant_range {
        pipeline_layout_info = pipeline_layout_info.push_constant_ranges(&push_constant_range);
    }
    let handle = unsafe { device.loader.create_pipeline_layout(&pipeline_layout_info, None) }?;

    Ok(handle)
}

fn get_or_create_pipeline_layout(
    device: &Device,
    cache: &mut PipelineLayoutCache,
    push_constant: Option<PushConstantBinding>,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> VkResult<vk::PipelineLayout> {
    let entry = PipelineLayoutEntry {
        push_constant,
        descriptor_set_layout,
    };
    cache
        .entry(entry)
        .try_or_insert_with(|| {
            create_pipeline_layout(
                device,
                push_constant
                    .as_ref()
                    .map(PushConstantBinding::push_constant_ranges)
                    .as_ref()
                    .map(core::slice::from_ref),
                Some(&[descriptor_set_layout]),
            )
        })
        .copied()
}

fn to_pipeline_table<P: Pipeline>(pipelines: Vec<P>) -> HashMap<Arc<str>, ArcFinalOwner<P>> {
    pipelines
        .into_iter()
        .map(|pipeline| (pipeline.cloned_id(), ArcFinalOwner::new(pipeline)))
        .collect()
}

#[derive(thiserror::Error, Debug)]
pub enum PipelineManagerError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    Ron(#[from] ron::de::SpannedError),
}

#[derive(Hash, Debug, PartialEq, Eq)]
pub struct PipelineLayoutEntry {
    pub push_constant: Option<PushConstantBinding>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub type PipelineLayoutCache = HashMap<PipelineLayoutEntry, vk::PipelineLayout>;
type ComputeDescriptorSetLayoutCache = HashMap<Vec<ComputeResourceType>, vk::DescriptorSetLayout>;
pub type ShaderModuleCache<'a, S> = HashMap<&'a str, ShaderModule<S>>;

// TODO: Pipeline cache: https://zeux.io/2019/07/17/serializing-pipeline-cache/.
// Could use an arena allocator for all the Arc IDs created here?.
pub struct PipelineManager {
    loader: SharedDeviceLoader,
    pub(crate) scene_set_layout: vk::DescriptorSetLayout,
    pipeline_layouts: PipelineLayoutCache,
    compute_descriptor_set_layouts: ComputeDescriptorSetLayoutCache,
    graphics_pipelines: HashMap<Arc<str>, ArcFinalOwner<GraphicsPipeline>>,
    compute_pipelines: HashMap<Arc<str>, ArcFinalOwner<ComputePipeline>>,
    bind_point_id_cache: HashSet<Arc<str>>,
}

impl PipelineManager {
    pub const SHADER_DIR: &'static str = "shaders/";
    pub const SHADER_PATH: &'static str = concatcp!(GalaxyEngine::CONTENT_PATH, PipelineManager::SHADER_DIR);
    pub const BUILT_SHADER_PATH: &'static str = concatcp!(GalaxyEngine::BUILT_PATH, PipelineManager::SHADER_DIR);
    const PIPELINE_CONFIG_GLOB: &'static str = "**/*.pipeline.ron";

    // TODO: Add support for game-specific pipeline configs.
    const ENGINE_PIPELINE_CONFIG_GLOB: &'static str =
        concatcp!(PipelineManager::SHADER_PATH, PipelineManager::PIPELINE_CONFIG_GLOB);

    pub const NUM_SCENE_DESCRIPTOR_SET_BINDINGS: usize = 5;

    fn scene_descriptor_set_layout_bindings(
    ) -> [vk::DescriptorSetLayoutBinding<'static>; Self::NUM_SCENE_DESCRIPTOR_SET_BINDINGS] {
        [
            // Scene uniforms:
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            // Transforms:
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            // Draw data:
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            // Material data storage:
            //vk::DescriptorSetLayoutBinding::default()
            //    .binding(3)
            //    .descriptor_count(1)
            //    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            //    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Material constant storage:
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Array of textures:
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_count(TextureManager::MAX_TEXTURES as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ]
    }

    pub fn new(device: &Device, msaa_samples: vk::SampleCountFlags) -> Result<Self, PipelineManagerError> {
        // Create scene descriptor set layout.
        let scene_set_layout = create_descriptor_set_layout(device, &Self::scene_descriptor_set_layout_bindings())?;

        // Find and load pipeline configs.
        let config_strings = glob(Self::ENGINE_PIPELINE_CONFIG_GLOB)
            .expect("Failed to read pipeline glob pattern")
            .filter_map(|path| {
                let unwrapped_path = match path.as_ref() {
                    Ok(path) => path,
                    Err(err) => err.path(),
                };
                let id = unwrapped_path.strip_prefix(Self::SHADER_PATH).ok()?;

                // A bunch of string allocations going on here. An arena would be nice.
                let mut id = PathBuf::from("/engine").join(id).to_slash()?.to_string();
                // Remove file extensions.
                if let Some(index) = id.find('.') {
                    id.truncate(index);
                }
                let id = Arc::<str>::from(id);

                // Nested function for error-handling.
                fn load_config(path: glob::GlobResult) -> Result<String, PipelineManagerError> {
                    Ok(std::fs::read_to_string(&path.map_err(|e| e.into_error())?)?)
                }

                let config_str = match load_config(path) {
                    Ok(config) => Some(config),
                    Err(err) => {
                        log::error!("Failed to read pipeline config for pipeline {id} ({err}).");
                        None
                    }
                }?;

                Some((id, config_str))
            })
            .collect::<Vec<_>>();

        let (graphics_configs, compute_configs): (Vec<_>, Vec<_>) = config_strings
            .iter()
            .filter_map(
                |(id, config_str)| match crate::utils::load_ron_config::<PipelineConfig>(config_str) {
                    Ok(config) => Some(config.with_id(id)),
                    Err(err) => {
                        log::error!("Failed to parse pipeline config for pipeline {id} ({err})");
                        None
                    }
                },
            )
            .partition_map(|config| match config {
                PipelineConfig::Graphics(graphics) => Either::Left(graphics),
                PipelineConfig::Compute(compute) => Either::Right(compute),
            });

        let mut pipeline_layouts = PipelineLayoutCache::new();

        // Compile graphics pipelines. For lots of graphics pipelines, could use a graphics pipeline library for speedup:
        // https://www.khronos.org/blog/reducing-draw-time-hitching-with-vk-ext-graphics-pipeline-library.
        let mut bind_point_id_cache = HashSet::new();
        let graphics_pipelines = {
            // Load shaders.
            let mut vertex_shaders = ShaderModuleCache::<shader_stage::Vertex>::new();
            let mut fragment_shaders = ShaderModuleCache::<shader_stage::Fragment>::new();
            for config in graphics_configs.iter() {
                vertex_shaders
                    .entry(config.shaders.vertex.id)
                    .try_or_insert_with(|| ShaderModule::new(device, config.shaders.vertex.id))?;
                fragment_shaders
                    .entry(config.shaders.fragment.id)
                    .try_or_insert_with(|| ShaderModule::new(device, config.shaders.fragment.id))?;
            }

            let create_resources = graphics_configs
                .into_iter()
                .map(|config| {
                    // Find or construct pipeline layout.
                    let pipeline_layout = get_or_create_pipeline_layout(
                        device,
                        &mut pipeline_layouts,
                        config.layout.push_constant,
                        scene_set_layout,
                    )?;

                    Ok(GraphicsPipelineCreateResources {
                        pipeline_layout,
                        vertex_shader: &vertex_shaders[config.shaders.vertex.id],
                        fragment_shader: &fragment_shaders[config.shaders.fragment.id],
                        config,
                    })
                })
                .collect::<VkResult<Vec<_>>>()?;

            log::info!("Compiling graphics pipelines...");
            let compilation_start = std::time::Instant::now();
            let graphics_pipelines =
                GraphicsPipeline::batch_new(device, create_resources, msaa_samples, &mut bind_point_id_cache)?;
            log::info!("Compiled graphics pipelines in {:?}", compilation_start.elapsed());
            graphics_pipelines
        };

        // Compile compute pipelines.
        let mut compute_descriptor_set_layouts = ComputeDescriptorSetLayoutCache::new();
        let compute_pipelines = {
            // Load shaders.
            let mut compute_shader_cache = ShaderModuleCache::<shader_stage::Compute>::new();
            for config in compute_configs.iter() {
                compute_shader_cache
                    .entry(config.shader)
                    .try_or_insert_with(|| ShaderModule::new(device, config.shader))?;
            }

            let create_resources = compute_configs
                .into_iter()
                .map(|config| {
                    // Get or create descriptor set layout.
                    let descriptor_set_layout = get_or_create_compute_descriptor_set_layout(
                        device,
                        &mut compute_descriptor_set_layouts,
                        config.layout.binding_types(),
                    )?;

                    // Construct pipeline layout.
                    let pipeline_layout = get_or_create_pipeline_layout(
                        device,
                        &mut pipeline_layouts,
                        config.layout.push_constant,
                        descriptor_set_layout,
                    )?;

                    Ok(ComputePipelineCreateResources {
                        pipeline_layout,
                        shader: &compute_shader_cache[config.shader],
                        config,
                    })
                })
                .collect::<VkResult<Vec<_>>>()?;

            log::info!("Compiling compute pipelines...");
            let compilation_start = std::time::Instant::now();
            let compute_pipelines = ComputePipeline::batch_new(device, create_resources)?;
            log::info!("Compiled compute pipelines in {:?}", compilation_start.elapsed());
            compute_pipelines
        };

        Ok(Self {
            loader: device.cloned_loader(),
            scene_set_layout,
            pipeline_layouts,
            compute_descriptor_set_layouts,
            graphics_pipelines: to_pipeline_table(graphics_pipelines),
            compute_pipelines: to_pipeline_table(compute_pipelines),
            bind_point_id_cache,
        })
    }

    pub fn get_bind_point_id(&self, id: &str) -> Option<&Arc<str>> {
        self.bind_point_id_cache.get(id)
    }

    pub fn get_graphics_pipeline(&self, id: &str) -> Option<&GraphicsPipeline> {
        self.graphics_pipelines.get(id).map(ArcFinalOwner::as_ref)
    }

    pub fn get_cloned_graphics_pipeline(&self, id: &str) -> Option<Arc<GraphicsPipeline>> {
        self.graphics_pipelines.get(id).map(Deref::deref).map(Arc::clone)
    }

    //pub fn iter_graphics_pipelines(&self) -> impl Iterator<Item = &GraphicsPipeline> {
    //    self.graphics_pipelines.values().map(ArcFinalOwner::as_ref)
    //}

    //pub fn num_layouts(&self) -> usize {
    //    self.pipeline_layouts.len()
    //}

    pub fn get_draw_layout(&self) -> Option<vk::PipelineLayout> {
        let entry = PipelineLayoutEntry {
            push_constant: Some(PushConstantBinding::DrawOffset),
            descriptor_set_layout: self.scene_set_layout,
        };
        self.pipeline_layouts.get(&entry).copied()
    }

    pub fn get_compute_descriptor_set_layout(
        &self,
        bindings: &[ComputeResourceType],
    ) -> Option<vk::DescriptorSetLayout> {
        self.compute_descriptor_set_layouts.get(bindings).copied()
    }

    pub fn get_compute_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.compute_pipelines.get(name).map(ArcFinalOwner::as_ref)
    }

    //pub fn get_layout(&self, entry: &PipelineLayoutEntry) -> Option<vk::PipelineLayout> {
    //    self.pipeline_layouts.get(entry).copied()
    //}
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

        // Destroy descriptor set layouts.
        for layout in self
            .compute_descriptor_set_layouts
            .values()
            .chain(std::iter::once(&self.scene_set_layout))
        {
            unsafe { self.loader.destroy_descriptor_set_layout(*layout, None) }
        }

        // Destroy pipeline layouts.
        for layout in self.pipeline_layouts.values() {
            unsafe { self.loader.destroy_pipeline_layout(*layout, None) }
        }
    }
}
