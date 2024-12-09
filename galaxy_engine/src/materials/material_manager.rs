// Copyright (c) 2024 Ben Sutherland.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;
use ultraviolet::{Vec2, Vec3, Vec4};

use crate::engine::GalaxyEngine;
use crate::materials::{Material, MaterialError, ResourceBinding};
use crate::pipelines::{GraphicsPipeline, Pipeline, PipelineBindingDataSize};
use crate::resource_paths::ResourcePath;
use crate::textures::TextureManager;
use crate::utils::LayoutExt;
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::gpu_alloc::MemResult;

pub struct IndexedMaterial {
    pub material: Arc<Material>,
    pub buffer_index: u32,
}

impl IndexedMaterial {
    pub fn new(material: Arc<Material>, index: u32) -> Self {
        Self {
            material,
            buffer_index: index,
        }
    }
}

pub struct LoadingMaterialManager {
    materials: HashMap<ResourcePath, IndexedMaterial>,
    pipelines: IndexMap<Arc<str>, Vec<ResourcePath>>,
}

impl LoadingMaterialManager {
    pub(crate) fn new() -> Self {
        Self {
            materials: HashMap::new(),
            pipelines: IndexMap::new(),
        }
    }

    pub fn get_or_load_material(
        &mut self,
        engine: &GalaxyEngine,
        texture_manager: &mut TextureManager,
        cmd_pool: &mut TransientPrimaryCommandPool,
        resource_path: &ResourcePath,
    ) -> Result<Arc<Material>, MaterialError> {
        if let Some(IndexedMaterial {
            buffer_index: _,
            material,
        }) = self.materials.get(resource_path)
        {
            Ok(Arc::clone(material))
        } else {
            let material = Arc::new(Material::new(engine, texture_manager, resource_path, cmd_pool)?);
            let resource_paths = self.pipelines.entry(material.pipeline().cloned_name()).or_default();
            let material_index = resource_paths.len() as u32;
            resource_paths.push(resource_path.clone());
            self.materials.insert(
                resource_path.clone(),
                IndexedMaterial::new(Arc::clone(&material), material_index),
            );
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
impl_material_binding!(Vec2, Vec2::new(1., 0.), [0]);
impl_material_binding!(Vec3, Vec3::new(1., 0., 1.), [0]);
impl_material_binding!(Vec4, Vec4::new(1., 0., 1., 1.), [0]);

struct PipelineData {
    materials: Vec<ResourcePath>,
    material_buffer: Buffer<GpuOnly>,
}

pub(crate) struct MaterialManager {
    materials: HashMap<ResourcePath, IndexedMaterial>,
    pipeline_data: IndexMap<Arc<str>, PipelineData>,
}

impl MaterialManager {
    pub fn new(
        loading: LoadingMaterialManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        // TODO: Upload all material data to the one buffer.
        let mut cmd_buf = cmd_pool.allocate_transient_cmd_buffer()?;
        let (_staging_buffers, pipeline_data) = loading
            .pipelines
            .into_iter()
            .map(|(name, resource_paths)| {
                let pipeline = engine.pipeline_manager.get_graphics_pipeline(&name).unwrap();
                let bindings_layout = pipeline.bindings_layout();
                let bindings_size = bindings_layout.layout.size();
                let (buffer_layout, padded_size) = bindings_layout.layout.repeat_pf(resource_paths.len()).unwrap();
                assert_eq!(
                    padded_size, bindings_size,
                    "Initial pad to align should ensure repeat does add any padding."
                );

                // Create material buffer. TODO: Make buffer take a Layout to pass alignment to the allocator.
                let mut material_buffer = Buffer::new(
                    debug_only_name!("{name} material buffer"),
                    &engine.device,
                    buffer_layout.size() as vk::DeviceSize,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    None,
                )?;

                // Copy material resource bindings to buffer.
                let staging_buffer =
                    material_buffer.copy_via_staging_buffer_with(&engine.device, &mut cmd_buf, None, |buffer| {
                        let buffer_memory = buffer.zero_and_get_mut_bytes();

                        for (i, path) in resource_paths.iter().enumerate() {
                            let indexed_material = &loading.materials[path];

                            let resource_bindings = &indexed_material.material.resource_bindings();
                            let buffer_range = i * bindings_size..(i + 1) * bindings_size;
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
                                            flags |= 1 << i;
                                        }
                                    }
                                    PipelineBindingDataSize::Float2 => {
                                        unimplemented!()
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
                            materials: resource_paths,
                            material_buffer,
                        },
                    ),
                ))
            })
            .collect::<MemResult<(Vec<_>, _)>>()?;

        // Upload material buffer.
        cmd_buf.end_submit_wait_and_free()?;

        Ok(Self {
            materials: loading.materials,
            pipeline_data,
        })
    }

    pub fn get_material_buffer_infos(&self) -> Vec<vk::DescriptorBufferInfo> {
        self.pipeline_data
            .values()
            .map(|data| {
                vk::DescriptorBufferInfo::default()
                    .buffer(data.material_buffer.handle())
                    .range(vk::WHOLE_SIZE)
            })
            .collect()
    }

    pub fn iter_materials_for_pipeline(&self, pipeline: &GraphicsPipeline) -> impl Iterator<Item = &IndexedMaterial> {
        self.pipeline_data[pipeline.name()]
            .materials
            .iter()
            .map(|resource_path| &self.materials[resource_path])
    }
}
