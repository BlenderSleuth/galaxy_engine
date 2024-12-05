// Copyright (c) 2024 Ben Sutherland.

use std::path::Path;
use std::slice;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;
use itertools::izip;
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, Ref, RefMut, View, ViewMut, World};

use crate::camera::{CamIsometry, Camera, FirstPersonCamera, ViewInfo};
use crate::engine::GalaxyEngine;
use crate::materials::{Material, MaterialData, MaterialError, MaterialResourceBinding};
use crate::meshes::mesh_manager::MeshManager;
use crate::meshes::{Mesh, MeshError};
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
        level: &mut Level,
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
        level: &mut Level,
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
        _level: &mut Level,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        Ok(())
    }
}

fn update_transform_with<F: FnOnce(&mut Transform)>(entity_id: EntityId, level: &mut Level, f: F) {
    level.world.run(|mut transforms: ViewMut<Transform>| {
        if let Some(mut transform) = transforms.get_or_insert(entity_id, Transform::default()) {
            f(&mut transform);
        } else {
            log::warn!("Failed to retrieve transform component for entity {entity_id:?}");
        }
    });
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
        level: &mut Level,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        update_transform_with(entity_id, level, |transform| {
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
        level: &mut Level,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        update_transform_with(entity_id, level, |transform| {
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
        level: &mut Level,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LoadError> {
        // Load material. TODO: Share materials.
        let material_path = ResourcePath::new(&self.material, Some(&level.config_path))
            .ok_or(LoadError::ResourcePathError(self.material.to_owned()))?;
        let material = Arc::new(Material::new(engine, &level.texture_manager, &material_path, cmd_pool)?);

        // Load mesh. TODO: Share meshes.
        let mesh_path = ResourcePath::new(&self.mesh, Some(&level.config_path))
            .ok_or(LoadError::ResourcePathError(self.mesh.to_owned()))?;
        let mesh = Arc::new(Mesh::new(
            Path::new(&self.mesh)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap_or("Unknown meshes"),
            engine,
            cmd_pool,
            &mesh_path,
        )?);

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
pub type SceneUniformBuffer = VolatileBuffer<SceneUniformData>;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DrawData {
    pub transform_index: u32,
    pub material_index: u32,
}

pub struct SceneBuffers {
    transforms: VolatileBuffer<[Mat4; 1024]>,
    materials: VolatileBuffer<[MaterialData; 1024]>,
}

impl SceneBuffers {
    pub fn iter_mut(&mut self, frame: usize) -> impl Iterator<Item = (&mut Mat4, &mut MaterialData)> {
        izip!(
            self.transforms.get_mut(frame).iter_mut(),
            self.materials.get_mut(frame).iter_mut()
        )
    }
    pub fn buffer_infos<const N: usize>(&self) -> [[vk::DescriptorBufferInfo; 2]; N] {
        core::array::from_fn(|frame| {
            [
                self.transforms.descriptor_buffer_info(frame),
                self.materials.descriptor_buffer_info(frame),
            ]
        })
    }
}

pub type SceneDescriptorPool = DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>;

pub struct Level {
    config_path: ResourcePath,
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: MeshManager,
    pub texture_manager: TextureManager,
    scene_descriptor_pool: SceneDescriptorPool,
    scene_uniform_buffer: SceneUniformBuffer,
    scene_buffers: SceneBuffers,
}

impl Level {
    pub fn new<T: DeserializableComponentConfig>(
        config_path: ResourcePath,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        _old_level: Option<Self>, // TODO: for reusing resources.
    ) -> Result<Self, LoadError> {
        // TODO: Unload previous level and resources.
        let device = &engine.device;

        // Create level uniform buffer.
        let scene_uniform_buffer = VolatileBuffer::new("Scene uniform buffer", device, VolatileBufferType::Uniform)?;

        let scene_transforms_buffer = VolatileBuffer::new("Transforms buffer", device, VolatileBufferType::Storage)?;
        // TODO: don't use volatile buffer for material data buffer.
        let scene_material_buffer = VolatileBuffer::new("Material buffer", device, VolatileBufferType::Storage)?;
        let scene_buffers = SceneBuffers {
            transforms: scene_transforms_buffer,
            materials: scene_material_buffer,
        };

        // Create level descriptor pool.
        let scene_descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * 3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((GalaxyEngine::MAX_FRAMES_IN_FLIGHT * GalaxyEngine::NUM_TEXTURES) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32),
        ];
        let mut scene_descriptor_pool =
            DescriptorPool::<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>::new(device, &scene_descriptor_pool_sizes)?;

        scene_descriptor_pool.allocate_descriptor_sets(
            device,
            &[engine.pipeline_manager.scene_descriptor_set_layout.handle(); GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        )?;

        let uniform_buffer_info: [_; GalaxyEngine::MAX_FRAMES_IN_FLIGHT] =
            core::array::from_fn(|frame| scene_uniform_buffer.descriptor_buffer_info(frame));
        let buffer_infos = scene_buffers.buffer_infos::<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>();

        let descriptor_writes: ArrayVec<_, { GalaxyEngine::MAX_FRAMES_IN_FLIGHT * 8 }> = scene_descriptor_pool
            .iter()
            .enumerate()
            .flat_map(|(frame, set)| {
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
                        .buffer_info(slice::from_ref(&buffer_infos[frame][0])),
                    // Material data:
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(slice::from_ref(&buffer_infos[frame][1])),
                ]
            })
            .collect();
        unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        let mut level = Self {
            world: World::new(),
            camera_entity: EntityId::dead(),
            mesh_manager: MeshManager::new(),
            texture_manager: TextureManager::new(&engine.device)?,
            scene_descriptor_pool,
            scene_uniform_buffer,
            scene_buffers,
            config_path,
        };

        // Parse config.
        let config_str = std::fs::read_to_string(&level.config_path.full_path::<resource_type::Level>(engine))?;
        let config = crate::utils::load_config::<LevelConfig<T>>(&config_str)?;

        for entity_config in config.entities {
            let id = level.world.add_entity(Name::new(entity_config.name));
            for component_config in entity_config.components.into_iter() {
                component_config.load(id, &mut level, engine, cmd_pool)?;
            }
        }

        level
            .texture_manager
            .write_textures_to_descriptor_array(engine, &level.scene_descriptor_pool);

        Ok(level)
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

        // Update meshes data.
        self.world.run(|v_models: View<Model>, v_transforms: View<Transform>| {
            (&v_models, &v_transforms)
                .iter()
                .zip(self.scene_buffers.iter_mut(frame_index))
                .for_each(|((model, transform), (transform_mat, material_data))| {
                    *transform_mat = view_info.mvp_from_transform(transform);

                    match model.material.resource_binding("base_colour").unwrap() {
                        MaterialResourceBinding::Texture(texture_index) => {
                            material_data.texture_index = *texture_index;
                        }
                    }
                });
        });
    }

    pub(crate) fn render(&self, rendering: &mut RenderingCmdBuf<PrimaryQueue>, frame_index: usize) {
        self.world.run(|v_models: View<Model>| {
            for model in v_models.iter() {
                model.material.bind(rendering);
                rendering.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    model.material.pipeline_layout(),
                    0,
                    slice::from_ref(&self.scene_descriptor_pool.get(frame_index)),
                    &[],
                );
                model.mesh.bind(rendering);
                model.mesh.draw(rendering);
            }
        });

        //self.particle_system.record_graphics(rendering, &view_info, time, viewport, scissor);
    }
}
