// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;
use std::slice;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, Ref, RefMut, View, ViewMut, World};

use crate::camera::{CamIsometry, Camera, FirstPersonCamera, ViewInfo};
use crate::engine::GalaxyEngine;
use crate::materials::{LoadingMaterialManager, Material, MaterialError, MaterialManager};
use crate::meshes::mesh_manager::MeshManager;
use crate::meshes::{Mesh, MeshError};
use crate::pipelines::{PipelineManager, PushConstantBinding};
use crate::prelude::*;
use crate::resource_paths::{resource_type, ResourcePath};
use crate::textures::TextureManager;
use crate::volatile_buffer::{VolatileBuffer, VolatileBufferType};
use crate::vulkan::command_buffer::{RenderingCmdBuf, TransientPrimaryCommandPool};
use crate::vulkan::descriptors::DescriptorPool;
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
    ) -> Result<(), LoadError>;
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
    ) -> Result<(), LoadError> {
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
    ) -> Result<(), LoadError> {
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
    index: BufferIndex,
}

impl ComponentConfig for Transform {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        level.world.add_component(
            entity_id,
            TransformComponent {
                transform: self.clone(),
                index: BufferIndex::default(),
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
    ) -> Result<(), LoadError> {
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
    ) -> Result<(), LoadError> {
        level.world.update_transform_with(entity_id, |transform| {
            transform.rotation = Rotor3::from_angle_plane(self.angle.to_radians(), self.plane.normalized());
        });
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelConfig {
    pub mesh: String,
    pub material: String,
}

#[derive(Component)]
pub struct Model {
    pub mesh: Arc<Mesh>,
    pub material: Arc<Material>,
}

impl ComponentConfig for ModelConfig {
    fn load(
        &self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        // Load material.
        let material_path = ResourcePath::new(&self.material, Some(&level.config_path))
            .ok_or(LoadError::ResourcePathError(self.material.clone()))?;
        let material = level.material_manager.get_or_load_material(
            engine,
            &mut level.texture_manager,
            cmd_pool,
            &material_path,
        )?;

        // Load mesh. TODO: Share meshes.
        let mesh_path = ResourcePath::new(&self.mesh, Some(&level.config_path))
            .ok_or(LoadError::ResourcePathError(self.mesh.clone()))?;
        let mesh_name = Path::new(&self.mesh)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap_or("Unknown mesh");
        let mesh = Arc::new(Mesh::new(mesh_name, engine, cmd_pool, &mesh_path)?);

        // Ensure each model has a transform.
        level.world.update_transform_with(entity_id, |_| {});

        level.world.add_component(entity_id, Model { mesh, material });
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
    pub mesh_manager: MeshManager,
    pub material_manager: LoadingMaterialManager,
    pub texture_manager: TextureManager,
}

//struct LevelDescriptorPool {
//    descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
//}
//
//impl LevelDescriptorPool {
//    // Two for the scene descriptor sets and one for the material descriptor set.
//    //const NUM_SETS: usize = GalaxyEngine::MAX_FRAMES_IN_FLIGHT + 1;
//
//    fn new(engine: &GalaxyEngine, level: &LoadingLevel) -> VkResult<Self> {
//
//        Ok(Self { descriptor_pool })
//    }
//
//    fn get(&self, frame_index: usize) -> vk::DescriptorSet {
//        self.descriptor_pool.get(frame_index)
//    }
//}

pub struct Level {
    _config_path: ResourcePath,
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: MeshManager,
    material_manager: MaterialManager,
    pub texture_manager: TextureManager,
    descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    scene_uniform_buffer: VolatileBuffer<SceneUniformData>,
    scene_transforms_buffer: VolatileBuffer<Mat4>,
}

impl Level {
    const MAX_TRANSFORMS: usize = 256;

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
            mesh_manager: MeshManager::new(),
            material_manager: LoadingMaterialManager::new(),
            texture_manager: TextureManager::new(&engine.device)?,
        };

        // Parse level config.
        let config_str = std::fs::read_to_string(level.config_path.full_path::<resource_type::Level>(engine))?;
        let config = crate::utils::load_config::<LevelConfig<T>>(&config_str)?;

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
            // Scene transforms buffer.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32),
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
        let scene_transforms_buffer = VolatileBuffer::new_array(
            "Scene transforms buffer",
            Level::MAX_TRANSFORMS,
            device,
            VolatileBufferType::Storage,
        )?;

        let material_manager = MaterialManager::new(level.material_manager, engine, cmd_pool)?;

        // Write to scene descriptor sets.
        let uniform_buffer_info = scene_uniform_buffer.descriptor_buffer_infos();
        let transform_buffer_info = scene_transforms_buffer.descriptor_buffer_infos();
        let texture_image_infos = level.texture_manager.get_image_infos();
        let material_buffer_infos = material_manager.get_material_buffer_addresses_infos();

        const NUM_WRITES: usize = 4;
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
                    // Material buffers.
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
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
                    .dst_binding(3) // Texture buffer is index 3 the in scene descriptor set layout.
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&texture_image_infos)
            }));
        }

        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        Ok(Self {
            _config_path: level.config_path,
            world: level.world,
            camera_entity: level.camera_entity,
            mesh_manager: level.mesh_manager,
            material_manager,
            texture_manager: level.texture_manager,
            descriptor_pool,
            scene_uniform_buffer,
            scene_transforms_buffer,
        })
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

        // Update transforms.
        self.world.run(|mut vm_transforms: ViewMut<TransformComponent>| {
            let transform_buffer = self.scene_transforms_buffer.get_mut_slice(frame_index);
            assert!(vm_transforms.len() <= transform_buffer.len());
            (&mut vm_transforms)
                .iter()
                .zip(transform_buffer.iter_mut())
                .enumerate()
                .for_each(|(i, (transform_comp, transform_mat))| {
                    *transform_mat = view_info.mvp_from_transform(&transform_comp.transform);
                    transform_comp.index.set(i as u32);
                });
        });
    }

    pub(crate) fn render(
        &self,
        pipeline_manager: &PipelineManager,
        cmd_buf: &mut RenderingCmdBuf<PrimaryQueue>,
        frame_index: usize,
    ) {
        // Get the drawing layout.
        let Some(layout) = pipeline_manager.get_layout(Some(PushConstantBinding::DrawData)) else {
            log::warn!("No pipelines to render with.");
            return;
        };

        // Bind scene descriptor set at index 0.
        cmd_buf.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            &[self.descriptor_pool.get(frame_index)],
            &[],
        );

        self.world
            .run(|v_models: View<Model>, v_transforms: View<TransformComponent>| {
                for (pipeline_index, pipeline) in pipeline_manager.iter_graphics_pipelines().enumerate() {
                    // Bind pipeline.
                    cmd_buf.bind_graphics_pipeline(pipeline);

                    for indexed_material in self.material_manager.iter_materials_for_pipeline(pipeline) {
                        // TODO: This is a stupid simple linear search to find models with the same material.
                        for (model, transform) in (&v_models, &v_transforms)
                            .iter()
                            .filter(|(model, _)| Arc::ptr_eq(&model.material, &indexed_material.material))
                        {
                            cmd_buf.push_constants(
                                layout,
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&DrawData {
                                    transform_index: transform.index.get().unwrap(),
                                    pipeline_index: pipeline_index as u32,
                                    material_index: indexed_material.buffer_index,
                                }),
                            );
                            model.mesh.bind(cmd_buf);
                            model.mesh.draw(cmd_buf);
                        }
                    }
                }
            });
    }
}
