// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use ultraviolet::{Vec2, Vec3, Vec4};

use super::config::MaterialConfigsCache;
use crate::engine::GalaxyEngine;
use crate::materials::{Material, MaterialError, ResourceBinding};
use crate::pipelines::{Pipeline, PipelineBindingDataSize};
use crate::resource_paths::SubresourcePath;
use crate::textures::TextureManager;
use crate::utils::LayoutExt;
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::gpu_alloc::MemResult;

pub struct LoadingMaterialManager {
    materials: HashMap<SubresourcePath, Arc<Material>>,
    pipelines: IndexMap<Arc<str>, Vec<SubresourcePath>>,
    configs: MaterialConfigsCache,
}

impl LoadingMaterialManager {
    pub(crate) fn new() -> Self {
        Self {
            materials: HashMap::new(),
            pipelines: IndexMap::new(),
            configs: MaterialConfigsCache::new(),
        }
    }

    pub fn get_or_load_material(
        &mut self,
        engine: &GalaxyEngine,
        texture_manager: &mut TextureManager,
        cmd_pool: &mut TransientPrimaryCommandPool,
        subresource_path: &SubresourcePath,
    ) -> Result<Arc<Material>, MaterialError> {
        if let Some(material) = self.materials.get(subresource_path) {
            Ok(Arc::clone(material))
        } else {
            let config = self.configs.get_or_load_material_config(engine, subresource_path)?;

            // The index in the material data buffer is the same as the index in the resource paths vec.
            let buffer_index = if self.pipelines.contains_key(config.pipeline) {
                self.pipelines[config.pipeline].len() as u32
            } else {
                0
            };
            let material = Arc::new(Material::new(
                engine,
                texture_manager,
                config,
                subresource_path.resource(),
                buffer_index,
                cmd_pool,
            )?);
            let resource_paths = if self.pipelines.contains_key(material.pipeline().id()) {
                &mut self.pipelines[material.pipeline().id()]
            } else {
                self.pipelines.entry(material.pipeline().cloned_id()).or_default()
            };
            resource_paths.push(subresource_path.clone());

            self.materials.insert(subresource_path.clone(), Arc::clone(&material));
            Ok(material)
        }
    }

    pub fn num_pipelines(&self) -> u32 {
        self.pipelines.len() as u32
    }
}

// Material binding trait and implementations.
trait MaterialBindingTrait<T: bytemuck::Pod> {
    const UNBOUND: T;
    fn get_texture_index(&self) -> u32;
    fn set_texture_index(&mut self, index: u32);
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct MaterialBinding<T> {
    constant: T,
}

macro_rules! impl_material_binding {
    ($ty:ty, $unbound:expr $(, $index:tt)?) => {
        impl MaterialBindingTrait<$ty> for MaterialBinding<$ty> {
            const UNBOUND: $ty = $unbound;
            fn get_texture_index(&self) -> u32 {
                self.constant$($index)?.to_bits()
            }
            fn set_texture_index(&mut self, index: u32) {
                self.constant$($index)? = f32::from_bits(index);
            }
        }

        impl fmt::Debug for MaterialBinding<$ty> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.debug_struct("MaterialBinding")
                    .field("texture_index", &self.get_texture_index())
                    .field("constant", &self.constant)
                    .finish()
            }
        }
    };
}

impl_material_binding!(f32, 0.5);
impl_material_binding!(Vec2, Vec2::new(0., 0.), [0]); // TODO: Make separate normal type with different unbound value.
impl_material_binding!(Vec3, Vec3::new(1., 0., 1.), [0]);
impl_material_binding!(Vec4, Vec4::new(1., 0., 1., 1.), [0]);

struct MaterialPathWithDataAddr {
    path: SubresourcePath,
    temp_arc: Arc<Material>,
    data_addr: vk::DeviceAddress,
}

struct PipelineData {
    materials: Vec<MaterialPathWithDataAddr>,
    _material_buffer: Buffer<GpuOnly>,
}

pub(crate) struct MaterialManager {
    pipeline_data: IndexMap<Arc<str>, PipelineData>,
    _materials: Vec<Arc<Material>>,
}

impl MaterialManager {
    pub fn new(
        loading: LoadingMaterialManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        let buffer_layouts = loading
            .pipelines
            .iter()
            .map(|(name, resource_paths)| {
                let pipeline = engine.pipeline_manager.get_graphics_pipeline(name).unwrap();
                let bindings_layout = pipeline.bindings_layout();
                let bindings_size = bindings_layout.layout.size();
                let (buffer_layout, padded_size) = bindings_layout.layout.repeat_pf(resource_paths.len()).unwrap();
                assert_eq!(
                    padded_size, bindings_size,
                    "Initial pad to align should ensure repeat does add any padding."
                );
                (
                    pipeline,
                    bindings_layout,
                    buffer_layout.size() as vk::DeviceSize,
                    bindings_size,
                )
            })
            .collect::<Vec<_>>();

        // TODO: Upload all material data to the one buffer.
        //let total_buffer_size = buffer_layouts.iter().map(|(_, _, size, _)| size).sum();

        let mut cmd_buf = cmd_pool.allocate_transient_cmd_buffer()?;
        let (_staging_buffers, pipeline_data) = loading
            .pipelines
            .into_iter()
            .zip(buffer_layouts)
            .map(
                |((name, resource_paths), (pipeline, bindings_layout, buffer_size, bindings_size))| {
                    // Create material buffer. TODO: Make buffer take a Layout to pass alignment to the allocator.
                    let mut material_buffer = Buffer::new(
                        debug_only_name!("{name} material buffer"),
                        &engine.device,
                        buffer_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )?;

                    let mut materials_paths = resource_paths
                        .into_iter()
                        .map(|path| MaterialPathWithDataAddr {
                            temp_arc: Arc::clone(&loading.materials[&path]),
                            path,
                            data_addr: material_buffer.device_address(), // Point to the start of the material data buffer.
                        })
                        .collect::<Vec<_>>();

                    // Copy material resource bindings to buffer.
                    let staging_buffer =
                        material_buffer.copy_via_staging_buffer_with(&engine.device, &mut cmd_buf, None, |buffer| {
                            let buffer_memory = buffer.zero_and_get_mut_bytes();

                            for (i, material_data) in materials_paths.iter_mut().enumerate() {
                                let material = &loading.materials[&material_data.path];
                                debug_assert_eq!(i, material.buffer_index() as usize);

                                let resource_bindings = &material.resource_bindings();
                                let buffer_range = i * bindings_size..(i + 1) * bindings_size;

                                // Offset the data address to point to the start of this material's data.
                                material_data.data_addr += buffer_range.start as vk::DeviceAddress;

                                let buffer_memory = &mut buffer_memory[buffer_range];

                                // A flags u32 is the final field, with a bit for each resource that says whether it's a constant (1) or a texture (0).
                                // Then union the first f32 of the constant as the texture index.
                                let mut flags = 0u32;
                                for (i, ((bind_point, binding), field_range)) in pipeline
                                    .bindings()
                                    .iter()
                                    .zip(bindings_layout.ranges.iter().cloned())
                                    .enumerate()
                                {
                                    // TODO: Remove code repetition here.
                                    match binding.ty {
                                        PipelineBindingDataSize::Float => {
                                            let mat_binding = bytemuck::from_bytes_mut::<MaterialBinding<f32>>(
                                                &mut buffer_memory[field_range],
                                            );
                                            if let Some(resource_binding) = resource_bindings.get(bind_point) {
                                                match resource_binding {
                                                    ResourceBinding::Constant(value) => {
                                                        mat_binding.constant = value.as_f32();
                                                        flags |= 1 << i;
                                                    }
                                                    ResourceBinding::Texture(index) => {
                                                        mat_binding.set_texture_index(*index);
                                                    }
                                                }
                                            } else {
                                                // TODO: In debug builds, put a constant sentinel value in the buffer to catch errors.
                                                mat_binding.constant = MaterialBinding::<f32>::UNBOUND;
                                                flags |= 1 << i;
                                            }
                                        }
                                        PipelineBindingDataSize::Float2 => {
                                            let mat_binding = bytemuck::from_bytes_mut::<MaterialBinding<Vec2>>(
                                                &mut buffer_memory[field_range],
                                            );
                                            if let Some(resource_binding) = resource_bindings.get(bind_point) {
                                                match resource_binding {
                                                    ResourceBinding::Constant(value) => {
                                                        mat_binding.constant = value.as_vec2();
                                                        flags |= 1 << i;
                                                    }
                                                    ResourceBinding::Texture(index) => {
                                                        mat_binding.set_texture_index(*index);
                                                    }
                                                }
                                            } else {
                                                mat_binding.constant = MaterialBinding::<Vec2>::UNBOUND;
                                                flags |= 1 << i;
                                            }
                                        }
                                        PipelineBindingDataSize::Float3 => {
                                            let mat_binding = bytemuck::from_bytes_mut::<MaterialBinding<Vec3>>(
                                                &mut buffer_memory[field_range],
                                            );
                                            if let Some(resource_binding) = resource_bindings.get(bind_point) {
                                                match resource_binding {
                                                    ResourceBinding::Constant(value) => {
                                                        mat_binding.constant = value.as_vec3();
                                                        flags |= 1 << i;
                                                    }
                                                    ResourceBinding::Texture(index) => {
                                                        mat_binding.set_texture_index(*index);
                                                    }
                                                }
                                            } else {
                                                mat_binding.constant = MaterialBinding::<Vec3>::UNBOUND;
                                                flags |= 1 << i;
                                            }
                                        }
                                        PipelineBindingDataSize::Float4 => {
                                            unimplemented!()
                                        }
                                        PipelineBindingDataSize::Normal => {
                                            unimplemented!()
                                        }
                                    }
                                }
                                let flags_range = bindings_layout.ranges.last().unwrap().clone();
                                *bytemuck::from_bytes_mut(&mut buffer_memory[flags_range]) = flags;

                                // Debug printing.
                                //#[repr(C)]
                                //#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
                                //struct UnlitMaterialData {
                                //    base_colour: MaterialBinding<Vec3>,
                                //    roughness: MaterialBinding<f32>,
                                //    normal: MaterialBinding<Vec3>,
                                //    flags: u32,
                                //}

                                //println!(
                                //    "Material data: {:?}",
                                //    bytemuck::from_bytes::<UnlitMaterialData>(
                                //        &buffer_memory[0..size_of::<UnlitMaterialData>()]
                                //    )
                                //);
                                //println!("Raw material data size: {:?}", buffer_memory.len());
                            }

                            Ok(())
                        })?;

                    Ok((
                        staging_buffer,
                        (
                            name,
                            PipelineData {
                                materials: materials_paths,
                                _material_buffer: material_buffer,
                            },
                        ),
                    ))
                },
            )
            .collect::<MemResult<(Vec<_>, IndexMap<_, _>)>>()?;

        // Upload material buffer.
        let pending = cmd_buf.end()?.submit(&[], &[])?;

        // Wait before dropping.
        pending.wait_for_fence()?;

        Ok(Self {
            _materials: loading.materials.into_values().collect(),
            pipeline_data,
        })
    }

    #[deprecated]
    pub fn get_material_data_buffer_addr(&self, material: &Arc<Material>) -> vk::DeviceAddress {
        self.pipeline_data[material.pipeline().id()]
            .materials
            .iter()
            .find(|data| Arc::ptr_eq(&data.temp_arc, material))
            .unwrap()
            .data_addr
    }

    //pub fn get_material_buffer_addresses_infos(
    //    &self,
    //) -> [vk::DescriptorBufferInfo; GalaxyEngine::MAX_FRAMES_IN_FLIGHT] {
    //    self.material_buffer_addresses.descriptor_buffer_infos()
    //}

    //pub fn iter_materials_for_pipeline(&self, pipeline: &GraphicsPipeline) -> impl Iterator<Item = &IndexedMaterial> {
    //    self.pipeline_data[pipeline.id()]
    //        .materials
    //        .iter()
    //        .map(|resource_path| &self.materials[resource_path])
    //}
}
