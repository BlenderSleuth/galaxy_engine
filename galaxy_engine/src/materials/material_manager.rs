// Copyright (c) 2024-2025 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use indexmap::IndexMap;

use super::config::MaterialConfigsCache;
use crate::engine::GalaxyEngine;
use crate::materials::{Material, MaterialError, ResourceBinding, ResourceConstant, ResourceRef};
use crate::pipelines::{Pipeline, PipelineBindingDataSize};
use crate::resource_paths::SubresourcePath;
use crate::textures::TextureManager;
use crate::vulkan::buffer::{Buffer, GpuOnly};
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::gpu_alloc::MemResult;

type ResourceConstantData = [u32; 4];

impl ResourceConstant {
    fn is_zero(&self) -> bool {
        match self {
            Self::Int(i) => *i == 0,
            Self::RGB(r, g, b) => *r == 0 && *g == 0 && *b == 0,
            Self::Float(f) => *f == 0.,
            Self::Float2(x, y) => *x == 0. && *y == 0.,
            Self::Float3(x, y, z) => *x == 0. && *y == 0. && *z == 0.,
            Self::Float4(x, y, z, w) => *x == 0. && *y == 0. && *z == 0. && *w == 0.,
        }
    }

    fn colour_component_encode(value: u8) -> u32 {
        ((value as f32) / 255.).to_bits()
    }

    fn write_constant(&self, constants_buf: &mut Vec<ResourceConstantData>) -> ResourceRef {
        // Special case for 0.
        if self.is_zero() {
            return ResourceRef::constant(0);
        }

        // Create resource ref.
        let resource = ResourceRef::constant(constants_buf.len() as u32);
        //log::info!("Resource ref = {}", resource.0);

        // Write the constant.
        constants_buf.push([0; 4]);
        let data = constants_buf.last_mut().unwrap();
        match self {
            Self::Int(i) => {
                data[0] = *i as u32;
            }
            Self::RGB(r, g, b) => {
                data[0] = Self::colour_component_encode(*r);
                data[1] = Self::colour_component_encode(*g);
                data[2] = Self::colour_component_encode(*b);
            }
            Self::Float(x) => {
                data[0] = x.to_bits();
            }
            Self::Float2(x, y) => {
                data[0] = x.to_bits();
                data[1] = y.to_bits();
            }
            Self::Float3(x, y, z) => {
                data[0] = x.to_bits();
                data[1] = y.to_bits();
                data[2] = z.to_bits();
            }
            Self::Float4(x, y, z, w) => {
                data[0] = x.to_bits();
                data[1] = y.to_bits();
                data[2] = z.to_bits();
                data[3] = w.to_bits();
            }
        };

        resource
    }
}

pub struct LoadingMaterialManager {
    resource_path_map: HashMap<SubresourcePath, u32>,
    materials: Vec<Arc<Material>>,
    pipelines: IndexMap<Arc<str>, Vec<SubresourcePath>>,
    configs: MaterialConfigsCache,
}

impl LoadingMaterialManager {
    pub const DEFAULT_MATERIAL: &'static str = "/engine/materials/default";

    pub(crate) fn new(
        engine: &GalaxyEngine,
        texture_manager: &mut TextureManager,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, MaterialError> {
        let mut material_manager = Self {
            resource_path_map: HashMap::new(),
            materials: Vec::new(),
            pipelines: IndexMap::new(),
            configs: MaterialConfigsCache::new(),
        };

        material_manager.get_or_load_material(
            engine,
            texture_manager,
            cmd_pool,
            SubresourcePath::new(Self::DEFAULT_MATERIAL, None).unwrap(),
        )?;

        Ok(material_manager)
    }

    pub fn get_or_load_material(
        &mut self,
        engine: &GalaxyEngine,
        texture_manager: &mut TextureManager,
        cmd_pool: &mut TransientPrimaryCommandPool,
        subresource_path: SubresourcePath,
    ) -> Result<Arc<Material>, MaterialError> {
        if let Some(material_index) = self.resource_path_map.get(&subresource_path) {
            Ok(Arc::clone(&self.materials[*material_index as usize]))
        } else {
            let config = self.configs.get_or_load_material_config(engine, &subresource_path)?;

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
                subresource_path.clone(),
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

            self.resource_path_map.insert(subresource_path, level_index);
            self.materials.push(Arc::clone(&material));
            Ok(material)
        }
    }

    pub fn num_pipelines(&self) -> u32 {
        self.pipelines.len() as u32
    }

    pub(crate) fn finalise_loading(
        self,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<MaterialManager> {
        MaterialManager::new(self, engine, cmd_pool)
    }
}

impl PipelineBindingDataSize {
    fn unbound(&self) -> &'static ResourceBinding {
        match self {
            Self::Float => &ResourceBinding::Constant(ResourceConstant::Float(0.)),
            Self::Float2 => &ResourceBinding::Constant(ResourceConstant::Float2(0., 0.)),
            Self::Float3 => &ResourceBinding::Constant(ResourceConstant::Float3(1., 0., 1.)),
            Self::Float4 => &ResourceBinding::Constant(ResourceConstant::Float4(1., 0., 1., 1.)),
            Self::Normal => &ResourceBinding::Constant(ResourceConstant::Float3(0., 0., 1.)),
        }
    }
}

pub struct MaterialManager {
    _materials: Vec<Arc<Material>>,
    _material_data_buffers: Vec<Buffer<GpuOnly>>,
    material_indices: Vec<u32>,
    material_data_addresses: Vec<vk::DeviceAddress>,
    //material_data_addresses_buffer: VolatileBuffer<vk::DeviceAddress, 1>,
    material_constants_buffer: Buffer<GpuOnly>,
}

impl MaterialManager {
    fn new(
        loading: LoadingMaterialManager,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<Self> {
        let materials = loading.materials;
        let mut material_constants: Vec<ResourceConstantData> = Vec::new();
        material_constants.push([0; 4]); // Zero value for the first constant.

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
            .map(|(pipeline_id, resource_paths)| {
                let pipeline = engine.pipeline_manager.get_graphics_pipeline(&pipeline_id).unwrap();
                let bindings_len = pipeline.bindings().len();
                let buffer_size = (resource_paths.len() * bindings_len * size_of::<ResourceRef>()) as vk::DeviceSize;

                // Create material buffer.
                let mut material_buffer = Buffer::new(
                    debug_only_name!("{pipeline_id} material data buffer"),
                    &engine.device,
                    buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )?;

                let mut materials_addrs: Vec<_> = resource_paths
                    .into_iter()
                    .map(|path| MaterialWithDataAddr {
                        mat: materials[loading.resource_path_map[&path] as usize].as_ref(),
                        data_addr: material_buffer.device_address(), // Point to the start of the material data buffer initially.
                    })
                    .collect();

                // Copy material resource bindings to buffer.
                let staging_buffer =
                    material_buffer.copy_via_staging_buffer_with(&engine.device, &mut cmd_buf, None, |buffer| {
                        let buffer_memory: &mut [ResourceRef] =
                            bytemuck::cast_slice_mut(buffer.zero_and_get_mut_bytes());

                        for (i, material_addr) in materials_addrs.iter_mut().enumerate() {
                            let buffer_range = i * bindings_len..(i + 1) * bindings_len;

                            // Offset the data address to point to the start of this material's data.
                            material_addr.data_addr +=
                                (buffer_range.start * size_of::<ResourceRef>()) as vk::DeviceAddress;

                            for ((bind_point, binding_size), resource_ref) in
                                pipeline.bindings().iter().zip(&mut buffer_memory[buffer_range])
                            {
                                let resource_binding = material_addr
                                    .mat
                                    .get_resource_binding(bind_point)
                                    .unwrap_or(binding_size.unbound());

                                *resource_ref = match resource_binding {
                                    ResourceBinding::Texture(index) => ResourceRef::texture(*index),
                                    ResourceBinding::Constant(constant) => {
                                        constant.write_constant(&mut material_constants)
                                    }
                                };
                            }
                        }

                        Ok(())
                    })?;

                Ok((staging_buffer, (material_buffer, materials_addrs)))
            })
            .collect::<MemResult<_>>()?;

        let mut material_constants_buffer = Buffer::new(
            "Material constants buffer",
            &engine.device,
            (material_constants.len() * size_of::<ResourceConstantData>()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let _constants_staging =
            material_constants_buffer.copy_via_staging_buffer_with(&engine.device, &mut cmd_buf, None, |buffer| {
                buffer.copy_slice_into_buffer(&material_constants, 0)?;
                Ok(())
            })?;

        // Upload material buffer.
        let pending = cmd_buf.end()?.submit(&[], &[])?;

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
            material_constants_buffer,
        })
    }

    pub(crate) fn material_constant_buffer_info(&self) -> vk::DescriptorBufferInfo {
        self.material_constants_buffer.descriptor_buffer_info()
    }

    pub fn get_material_index(&self, material: &Material) -> u32 {
        self.material_indices[material.level_index() as usize]
    }

    pub(crate) fn get_material_addr(&self, material: &Material) -> vk::DeviceAddress {
        self.material_data_addresses[self.get_material_index(material) as usize]
    }
}
