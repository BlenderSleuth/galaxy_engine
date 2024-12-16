// Copyright (c) 2024 Ben Sutherland.

use std::slice;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;
use itertools::izip;
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, Ref, RefMut, View, ViewMut, World};

use crate::camera::{CamIsometry, Camera, FirstPersonCamera, ViewInfo};
use crate::engine::GalaxyEngine;
use crate::materials::{LoadingMaterialManager, Material, MaterialError, MaterialManager};
use crate::meshes::mesh_manager::{LoadingMeshManager, MeshManager};
use crate::meshes::{Mesh, MeshError};
use crate::pipelines::PipelineManager;
use crate::prelude::*;
use crate::resource_paths::{resource_type, ResourcePath, SubresourcePath};
use crate::textures::TextureManager;
use crate::volatile_buffer::{VolatileBuffer, VolatileBufferType};
use crate::vulkan::command_buffer::{RenderingCmdBuf, TransientPrimaryCommandPool};
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemoryError;
use crate::vulkan::queue::queue_type::PrimaryQueue;

#[enum_delegate::register]
pub trait ComponentConfig {
    fn load(
        &self, // use self: Box<Self>, if going the boxed route with a custom deserialiser.
        entity_id: EntityId,
        level: &mut LoadingLevel,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()>;
}

// Allow adding configs that are components themselves directly to the world.
impl<T> ComponentConfig for T
where
    T: Component + Clone + Serialize + for<'a> Deserialize<'a>,
{
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.add_component(entity_id, self.clone());
        Ok(())
    }
}

pub trait DeserializableComponentConfig: ComponentConfig + for<'a> Deserialize<'a> {}
impl<T> DeserializableComponentConfig for T where T: ComponentConfig + for<'a> Deserialize<'a> {}

#[derive(Serialize, Deserialize, Debug)]
pub struct LightConfig {
    pub colour: Vec3,
    pub intensity: f32,
}

impl ComponentConfig for LightConfig {
    fn load(
        &self,
        _entity_id: EntityId,
        _level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct BufferIndex(u32);

impl Default for BufferIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl BufferIndex {
    const INVALID: Self = Self(u32::MAX);

    pub fn is_valid(&self) -> bool {
        self.0 != Self::INVALID.0
    }

    pub fn get(&self) -> Option<u32> {
        if self.is_valid() {
            Some(self.0)
        } else {
            None
        }
    }

    pub fn set(&mut self, index: u32) {
        self.0 = index;
    }
}

#[derive(Component, Default)]
pub struct TransformComponent {
    transform: Transform,
}

impl ComponentConfig for Transform {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.add_component(
            entity_id,
            TransformComponent {
                transform: self.clone(),
            },
        );
        Ok(())
    }
}

pub trait WorldExt {
    fn update_transform_with<F: FnOnce(&mut Transform)>(&self, entity_id: EntityId, f: F);
}

impl WorldExt for World {
    fn update_transform_with<F: FnOnce(&mut Transform)>(&self, entity_id: EntityId, f: F) {
        self.run(|mut transforms: ViewMut<TransformComponent>| {
            if let Some(mut component) = transforms.get_or_insert(entity_id, TransformComponent::default()) {
                f(&mut component.transform);
            } else {
                log::warn!("Failed to retrieve transform component for entity {entity_id:?}");
            }
        });
    }
}

#[derive(Serialize, Deserialize, Component, Debug, Clone)]
#[serde(transparent)]
pub struct IsometryComponent(pub(crate) Isometry3);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Scale(f32);

impl ComponentConfig for Scale {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.update_transform_with(entity_id, |transform| {
            transform.scale = Vec3::broadcast(self.0);
        });
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnglePlaneRotor {
    pub angle: f32, // Degrees
    pub plane: Bivec3,
}

impl ComponentConfig for AnglePlaneRotor {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.update_transform_with(entity_id, |transform| {
            transform.rotation = Rotor3::from_angle_plane(self.angle.to_radians(), self.plane.normalized());
        });
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelConfig {
    pub mesh: String,
    pub materials: Vec<String>,
}

#[derive(Component)]
pub struct Model {
    // mesh.num_elements() == materials.len()
    pub mesh: Arc<Mesh>,
    pub materials: Vec<Arc<Material>>,
    pub draw_index: Option<u32>,
}

impl ComponentConfig for ModelConfig {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        if self.materials.is_empty() {
            return Ok(()); // No materials, nothing to do.
        }

        // Load mesh.
        let mesh_path = ResourcePath::new(&self.mesh, Some(&level.config_path))
            .ok_or(LoadError::ResourcePathError(self.mesh.clone()))?;
        let mesh = level.mesh_manager.get_or_load_mesh(engine, cmd_pool, &mesh_path)?;

        // Load materials.
        let mut materials = self
            .materials
            .iter()
            .map(|material| {
                let material_path = SubresourcePath::new(material, Some(&level.config_path))
                    .ok_or(LoadError::ResourcePathError(material.clone()))?;
                Ok(level.material_manager.get_or_load_material(
                    engine,
                    &mut level.texture_manager,
                    cmd_pool,
                    &material_path,
                )?)
            })
            .collect::<LoadResult<Vec<_>>>()?;

        let num_elements = mesh.num_elements() as usize;
        match materials.len().cmp(&num_elements) {
            std::cmp::Ordering::Less => {
                log::warn!("Too few materials for mesh, duplicating last material.");
                let last_material = Arc::clone(materials.last().unwrap());
                materials.resize(num_elements, last_material);
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                log::warn!("Too many materials for mesh, truncating to {num_elements}.");
                materials.truncate(num_elements);
            }
        }

        debug_assert_eq!(materials.len(), mesh.num_elements() as usize);

        // Ensure each model has a transform.
        level.world.update_transform_with(entity_id, |_| {});

        level.world.add_component(
            entity_id,
            Model {
                mesh,
                materials,
                draw_index: None,
            },
        );
        Ok(())
    }
}

// TODO: This is a hack to avoid needing to write a custom deserialiser for the enum.
// Do this properly with a custom deserialiser.
#[macro_export]
macro_rules! register_components {
    ($enum_name:ident, $($name:ident: $config:ident),*) => {
        #[derive(Serialize, Deserialize, Debug)]
        #[enum_delegate::implement(ComponentConfig)]
        pub enum $enum_name {
            Camera(galaxy_engine::camera::CameraConfig),
            Light(galaxy_engine::level::LightConfig),
            Transform(galaxy_engine::maths::Transform),
            Isometry(galaxy_engine::level::IsometryComponent),
            Scale(galaxy_engine::level::Scale),
            AnglePlaneRotor(galaxy_engine::level::AnglePlaneRotor),
            Model(galaxy_engine::level::ModelConfig),
            $($name($config)),*
        }
    };
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename = "Entity")]
pub struct EntityConfig<'a, T> {
    pub name: &'a str,
    pub components: Vec<T>,
}

#[derive(Component)]
pub struct Name {
    name: Box<str>,
}

impl Name {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned().into_boxed_str(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename = "Level")]
pub struct LevelConfig<'a, T> {
    #[serde(borrow = "'a")]
    pub entities: Vec<EntityConfig<'a, T>>,
}

#[derive(thiserror::Error, Debug)]
pub enum LoadError {
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Mesh vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
    #[error("Material error: {0}")]
    MaterialError(#[from] MaterialError),
    #[error("Mesh error: {0}")]
    MeshError(#[from] MeshError),
    #[error("Resource path could not be resolved: {0}")]
    ResourcePathError(String),
    #[error("No camera component found")]
    NoCameraComponent,
}

pub type LoadResult<T> = Result<T, LoadError>;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SceneUniformData {
    view: Mat4,
    proj: Mat4,
    sun_direction: Vec3,
    delta_time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DrawData {
    pub transform_index: u32,
    pub pipeline_index: u32,
    pub material_index: u32,
}

pub struct LoadingLevel {
    config_path: ResourcePath,
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: LoadingMeshManager,
    pub material_manager: LoadingMaterialManager,
    pub texture_manager: TextureManager,
}

pub struct Level {
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: MeshManager,
    material_manager: MaterialManager,
    pub texture_manager: TextureManager,
    scene_descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    scene_uniform_buffer: VolatileBuffer<SceneUniformData>,
    scene_transforms_buffer: VolatileBuffer<Mat4>,
    element_offsets: VolatileBuffer<u32>,
    material_data_addresses: VolatileBuffer<vk::DeviceAddress>,
    draw_indirect_buffer: VolatileBuffer<crate::pod::vk::DrawIndexedIndirectCommand>,
}

impl Level {
    pub fn new<T: DeserializableComponentConfig>(
        config_path: ResourcePath,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        _old_level: Option<Self>, // TODO: for reusing resources.
    ) -> Result<Self, LoadError> {
        // TODO: Unload previous level and resources.

        let mut level = LoadingLevel {
            config_path,
            world: World::new(),
            camera_entity: EntityId::dead(),
            mesh_manager: LoadingMeshManager::new(),
            material_manager: LoadingMaterialManager::new(),
            texture_manager: TextureManager::new(&engine.device)?,
        };

        // Parse level config.
        let config_str = std::fs::read_to_string(level.config_path.full_path::<resource_type::Level>(engine))?;
        let config = crate::utils::load_ron_config::<LevelConfig<T>>(&config_str)?;

        // Load level.
        for entity_config in config.entities {
            let id = level.world.add_entity(Name::new(entity_config.name));
            for component_config in entity_config.components.into_iter() {
                component_config.load(id, &mut level, engine, cmd_pool)?;
            }
        }

        let device = &engine.device;

        // Create descriptor pool.
        let descriptor_pool_sizes = [
            // Scene uniform buffer.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32),
            // Scene transforms + elements offsets buffers.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * 2),
            // Scene texture descriptor array.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(
                    (GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * level.texture_manager.num_textures()).max(1),
                ),
            // Material data buffers. TODO: When we have more than one incompatible pipeline layout, allocate pipeline material data buffers per layout.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * level.material_manager.num_pipelines()),
        ];
        let mut descriptor_pool = DescriptorPool::new(&engine.device, &descriptor_pool_sizes)?;

        descriptor_pool.allocate_descriptor_sets(
            &engine.device,
            &[engine.pipeline_manager.scene_set_layout; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        )?;

        // Create scene buffers.
        let scene_uniform_buffer = VolatileBuffer::new("Scene uniform buffer", device, VolatileBufferType::Uniform)?;

        let num_models = level.world.iter::<&Model>().iter().count();
        let scene_transforms_buffer = VolatileBuffer::new_array(
            "Scene transforms buffer",
            num_models,
            device,
            VolatileBufferType::Storage,
        )?;

        // Finish material and mesh loading.
        let material_manager = MaterialManager::new(level.material_manager, engine, cmd_pool)?;
        let mesh_manager = MeshManager::new(level.mesh_manager, engine, cmd_pool)?;

        let total_scene_material_refs = level
            .world
            .iter::<&Model>()
            .iter()
            .map(|model| model.materials.len())
            .sum();
        let element_offsets_buffer = VolatileBuffer::new_array(
            "Element offsets buffer",
            num_models,
            device,
            VolatileBufferType::Storage,
        )?;
        let material_buffer_addresses = VolatileBuffer::new_array(
            "Material buffer addresses",
            total_scene_material_refs,
            &engine.device,
            VolatileBufferType::Storage,
        )?;

        // Write to scene descriptor sets.
        let uniform_buffer_info = scene_uniform_buffer.descriptor_buffer_infos();
        let transform_buffer_info = scene_transforms_buffer.descriptor_buffer_infos();
        let element_offset_buffer_info = element_offsets_buffer.descriptor_buffer_infos();
        let texture_image_infos = level.texture_manager.get_image_infos();
        let material_buffer_infos = material_buffer_addresses.descriptor_buffer_infos();

        const NUM_WRITES: usize = 5;
        let mut descriptor_writes: ArrayVec<_, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT * NUM_WRITES }> = descriptor_pool
            .iter()
            .enumerate()
            .flat_map(|(frame, set)| -> [vk::WriteDescriptorSet; NUM_WRITES - 1] {
                [
                    // Uniform buffer:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(slice::from_ref(&uniform_buffer_info[frame])),
                    // Transforms buffer:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&transform_buffer_info[frame])),
                    // Element offset buffer.
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&element_offset_buffer_info[frame])),
                    // Material buffers.
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&material_buffer_infos[frame])),
                ]
            })
            .collect();

        if !texture_image_infos.is_empty() {
            descriptor_writes.extend(descriptor_pool.iter().map(|set| {
                // Textures array.
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(4) // Texture buffer is index 3 the in scene descriptor set layout.
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&texture_image_infos)
            }));
        }

        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        // Create draw indirect buffer.
        //let draw_indirect_commands_size = (size_of::<vk::DrawIndexedIndirectCommand>() * num_models) as vk::DeviceSize;
        let draw_indirect_buffer = VolatileBuffer::new_array(
            "Draw indirect buffer",
            num_models,
            &engine.device,
            VolatileBufferType::Indirect,
        )?;
        //let mut draw_indirect_buffer_info = draw_indirect_buffer.descriptor_buffer_info();
        //draw_indirect_buffer_info.range = draw_indirect_commands_size;
        //let indirect_draw_count_addr = draw_indirect_buffer.device_address() + draw_indirect_commands_size;

        Ok(Self {
            world: level.world,
            camera_entity: level.camera_entity,
            mesh_manager,
            material_manager,
            texture_manager: level.texture_manager,
            scene_descriptor_pool: descriptor_pool,
            scene_uniform_buffer,
            scene_transforms_buffer,
            material_data_addresses: material_buffer_addresses,
            element_offsets: element_offsets_buffer,
            draw_indirect_buffer,
        })
    }

    pub(crate) fn notify_window_resize(&self, width: u32, height: u32) {
        // Update camera aspect.
        let mut camera = self.world.get::<&mut Camera>(self.camera_entity).unwrap();
        camera.aspect = width as f32 / height as f32;
    }

    pub fn get_camera(&self) -> (Ref<&Isometry3>, Ref<&Camera>) {
        let (cam_transform, cam) = self
            .world
            .get::<(&IsometryComponent, &Camera)>(self.camera_entity)
            .unwrap();
        (Ref::map(cam_transform, |t| &t.0), cam)
    }

    pub fn get_camera_transform_mut(&mut self) -> RefMut<&mut Isometry3> {
        let cam_transform = self.world.get::<&mut IsometryComponent>(self.camera_entity).unwrap();
        RefMut::map(cam_transform, |t| &mut t.0)
    }

    // Called before the gpu fence, for cpu-only updates this frame.
    pub(crate) fn update(&mut self, engine: &GalaxyEngine, delta_time: f32, mouse_delta: Vec2) {
        // Update camera.
        {
            let mut cam_transform = self.get_camera_transform_mut();

            // Update camera rotation.
            {
                const ROTATE_SPEED: f32 = 0.1;
                let first_person_mouse = -mouse_delta * ROTATE_SPEED;
                cam_transform.as_mut().apply_first_person_mouse(first_person_mouse);
            }

            // Update camera position.
            {
                const MOVE_SPEED: f32 = 3.;

                let mut camera_velocity = Vec3::zero();

                if engine.is_key_pressed("w") {
                    camera_velocity += cam_transform.cam_forward();
                }
                if engine.is_key_pressed("s") {
                    camera_velocity -= cam_transform.cam_forward();
                }

                if engine.is_key_pressed("a") {
                    camera_velocity -= cam_transform.cam_right();
                }

                if engine.is_key_pressed("d") {
                    camera_velocity += cam_transform.cam_right();
                }

                if engine.is_key_pressed("e") {
                    camera_velocity += Vec3::unit_z();
                }

                if engine.is_key_pressed("q") {
                    camera_velocity -= Vec3::unit_z();
                }

                if camera_velocity.mag_sq() > 1e-6 {
                    camera_velocity.normalize();
                }

                cam_transform.translation += camera_velocity * MOVE_SPEED * delta_time;
            }
        }
    }

    // Called after the gpu fence, so gpu buffers can be updated.
    pub(crate) fn gpu_update(&mut self, delta_time: f32, game_time: std::time::Duration, frame_index: usize) {
        let view_info = {
            let (cam_transform, cam) = self.get_camera();
            ViewInfo::new(&cam, &cam_transform)
        };

        let time = game_time.as_secs_f64();

        // Update GPU buffers.

        // Update scene uniforms.
        *self.scene_uniform_buffer.get_mut(frame_index) = SceneUniformData {
            view: view_info.view,
            proj: view_info.projection,
            sun_direction: Vec3::new(
                time.sin().abs() as f32,
                (time + 0.3).sin().abs() as f32,
                (time + 0.6).sin().abs() as f32,
            ),
            delta_time,
        };

        //{
        //    let addresses = material_buffer_addresses.get_mut_slice(0);
        //    for (data, address) in pipeline_data.values().zip(addresses) {
        //        *address = data.material_buffer.device_address();
        //    }
        //}

        //for _frame in 1..GalaxyEngine::MAX_FRAMES_IN_FLIGHT {
        //    // Copy first frame to second.
        //}

        // Update scene data.
        self.world
            .run(|v_models: View<Model>, v_transforms: View<TransformComponent>| {
                let transform_buffer = self.scene_transforms_buffer.get_mut_slice(frame_index);
                let element_offsets = self.element_offsets.get_mut_slice(frame_index);
                let material_data_addresses = self.material_data_addresses.get_mut_slice(frame_index);
                let draw_indirect_buffer = self.draw_indirect_buffer.get_mut_slice(frame_index);

                debug_assert!(v_models.len() <= transform_buffer.len());

                let mut current_element_offset = 0;
                izip!(
                    (&v_transforms, &v_models).iter(),
                    transform_buffer.iter_mut(),
                    element_offsets.iter_mut(),
                    draw_indirect_buffer.iter_mut(),
                )
                .for_each(
                    |((transform_comp, model), transform_mat, element_offset, draw_indirect)| {
                        // Write transform.
                        *transform_mat = view_info.mvp_from_transform(&transform_comp.transform);
                        // Write element offset.
                        *element_offset = current_element_offset;
                        // Write draw indirect command.
                        let draw_params = self.mesh_manager.draw_command_for_mesh(&model.mesh);
                        draw_indirect.index_count = model.mesh.num_indices();
                        draw_indirect.instance_count = 1;
                        draw_indirect.first_index = draw_params.index_offset;
                        draw_indirect.vertex_offset = draw_params.vertex_offset;
                        draw_indirect.first_instance = 0;

                        // Write material data.
                        for (i, material) in model.materials.iter().enumerate() {
                            material_data_addresses[current_element_offset as usize + i] =
                                self.material_manager.get_material_data_buffer_addr(material);
                        }
                        current_element_offset += model.mesh.num_elements();
                    },
                );
            });
    }

    pub(crate) fn render(
        &self,
        device: &Device,
        pipeline_manager: &PipelineManager,
        cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>,
        frame_index: usize,
    ) {
        // Get the drawing pipeline layout.
        let Some(layout) = pipeline_manager.get_draw_layout() else {
            log::warn!("No pipelines to draw with.");
            return;
        };

        // Bind scene descriptor set at index 0.
        cmd_buf.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            &[self.scene_descriptor_pool.get(frame_index)],
            &[],
        );

        // TODO: Num models per pipeline.
        let num_models = self.world.iter::<&Model>().iter().count();

        for pipeline in pipeline_manager.iter_graphics_pipelines() {
            // Bind pipeline.
            cmd_buf.bind_graphics_pipeline(pipeline);
            self.mesh_manager.bind_megabuffer(cmd_buf);
            unsafe {
                device.loader().cmd_draw_indexed_indirect(
                    cmd_buf.handle_dep(),
                    self.draw_indirect_buffer.handle_dep(),
                    self.draw_indirect_buffer.frame_offset(frame_index) as vk::DeviceSize,
                    num_models as u32,
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                )
            };
        }
    }
}
