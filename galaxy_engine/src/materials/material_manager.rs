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
use crate::volatile_buffer::{VolatileBuffer, VolatileBufferType};
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::gpu_alloc::MemResult;

pub struct LoadingMaterialManager {
    resource_path_map: HashMap<SubresourcePath, u32>,
    materials: Vec<Arc<Material>>,
    pipelines: IndexMap<Arc<str>, Vec<SubresourcePath>>,
    configs: MaterialConfigsCache,
}

impl LoadingMaterialManager {
    pub(crate) fn new() -> Self {
        // TODO: Upload debug error material to index 0.
        Self {
            resource_path_map: HashMap::new(),
            materials: Vec::new(),
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
        if let Some(material_index) = self.resource_path_map.get(subresource_path) {
            Ok(Arc::clone(&self.materials[*material_index as usize]))
        } else {
            let config = self.configs.get_or_load_material_config(engine, subresource_path)?;

            // The index in the material data buffer is the same as the index in the resource paths vec.
            //let buffer_index = if self.pipelines.contains_key(config.pipeline) {
            //    self.pipelines[config.pipeline].len() as u32
            //} else {
            //    0
            //};
            let level_index = self.materials.len() as u32;
            let material = Arc::new(Material::new(
                engine,
                texture_manager,
                config,
                subresource_path.resource(),
                level_index,
                //buffer_index,
                cmd_pool,
            )?);
            let resource_paths = if self.pipelines.contains_key(material.pipeline().id()) {
                &mut self.pipelines[material.pipeline().id()]
            } else {
                self.pipelines.entry(material.pipeline().cloned_id()).or_default()
            };
            resource_paths.push(subresource_path.clone());

            self.resource_path_map.insert(subresource_path.clone(), level_index);
            self.materials.push(Arc::clone(&material));
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

pub struct MaterialManager {
    _materials: Vec<Arc<Material>>,
    _material_data_buffers: Vec<Buffer<GpuOnly>>,
    material_indices: Vec<u32>,
    material_data_addresses: Vec<vk::DeviceAddress>,
    material_data_addresses_buffer: VolatileBuffer<vk::DeviceAddress, 1>,
}

impl MaterialManager {
    pub(crate) fn new(
        loading: LoadingMaterialManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        let materials = loading.materials;

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

        struct MaterialWithDataAddr<'a> {
            mat: &'a Material,
            data_addr: vk::DeviceAddress,
        }

        let mut cmd_buf = cmd_pool.allocate_transient_cmd_buffer()?;
        let (_staging_buffers, (material_data_buffers, pipeline_data)): (Vec<_>, (Vec<_>, Vec<_>)) = loading
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

                    let mut materials_addrs: Vec<_> = resource_paths
                        .into_iter()
                        .map(|path| MaterialWithDataAddr {
                            mat: materials[loading.resource_path_map[&path] as usize].as_ref(),
                            data_addr: material_buffer.device_address(), // Point to the start of the material data buffer.
                        })
                        .collect();

                    // Copy material resource bindings to buffer.
                    let staging_buffer =
                        material_buffer.copy_via_staging_buffer_with(&engine.device, &mut cmd_buf, None, |buffer| {
                            let buffer_memory = buffer.zero_and_get_mut_bytes();

                            for (i, material_data) in materials_addrs.iter_mut().enumerate() {
                                let material = material_data.mat;
                                //debug_assert_eq!(i, material.buffer_index() as usize);

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

                    Ok((staging_buffer, (material_buffer, materials_addrs)))
                },
            )
            .collect::<MemResult<_>>()?;

        // Upload material buffer.
        let pending = cmd_buf.end()?.submit(&[], &[])?;

        // Set up material data addresses buffer.
        let material_data_addresses_buffer = VolatileBuffer::new_array(
            "Material data addresses",
            materials.len(),
            &engine.device,
            VolatileBufferType::Storage,
        )?;

        // Set material data addresses and record where the material data address is in the buffer.
        // It's in a different order because materials are grouped in the buffer by pipeline, rather than upload order.
        let mut material_indices = vec![0; materials.len()];
        let material_data_addresses = pipeline_data
            .iter()
            .flat_map(|data| data.iter().map(|data| (data.data_addr, data.mat)))
            .enumerate()
            .map(|(i, (data_addr, material))| {
                // Build the material index map.
                material_indices[material.level_index() as usize] = i as u32;
                // Collect the material data addresses.
                data_addr
            })
            .collect();

        // Wait before dropping.
        pending.wait_for_fence()?;

        Ok(Self {
            _materials: materials,
            _material_data_buffers: material_data_buffers,
            material_indices,
            material_data_addresses,
            material_data_addresses_buffer,
        })
    }

    #[deprecated]
    pub(crate) fn material_data_addresses_info(&self) -> vk::DescriptorBufferInfo {
        self.material_data_addresses_buffer.descriptor_buffer_info(0)
    }

    pub fn get_material_index(&self, material: &Material) -> u32 {
        self.material_indices[material.level_index() as usize]
    }

    pub(crate) fn get_material_addr(&self, material: &Material) -> vk::DeviceAddress {
        self.material_data_addresses[self.get_material_index(material) as usize]
    }
}
