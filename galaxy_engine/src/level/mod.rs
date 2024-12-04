// Copyright (c) 2024 Ben Sutherland.

use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, IntoWithId, Ref, RefMut, View, World};
use ultraviolet::{Isometry3, Rotor3, Vec3};

use crate::camera::Camera;
use crate::engine::GalaxyEngine;
use crate::materials::{Material, MaterialError};
use crate::mesh::{Mesh, MeshError};
use crate::textures::TextureError;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;
use crate::vulkan::gpu_alloc::MemoryError;

#[enum_delegate::register]
pub trait ComponentConfig {
    fn load(
        &self,
        entity_id: EntityId,
        world: &mut World,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LevelLoadError>;
}

// Allow adding configs that are components themselves directly.
impl<T> ComponentConfig for T
where
    T: Component + Clone,
{
    fn load(
        &self,
        entity_id: EntityId,
        world: &mut World,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LevelLoadError> {
        world.add_component(entity_id, self.clone());
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
        _world: &mut World,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LevelLoadError> {
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Component, Debug, Clone)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Rotor3,
    pub scale: Vec3,
}

#[derive(Serialize, Deserialize, Component, Debug, Clone)]
#[serde(transparent)]
pub struct IsometryComponent(pub(crate) Isometry3);

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelConfig {
    pub mesh: String,
    pub material: String,
}

#[derive(Component)]
pub struct Model {
    mesh: Arc<Mesh>,
    material: Arc<Material>,
}

impl ComponentConfig for ModelConfig {
    fn load(
        &self,
        _entity_id: EntityId,
        _world: &mut World,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<(), LevelLoadError> {
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
            Transform(galaxy_engine::level::Transform),
            Isometry(galaxy_engine::level::IsometryComponent),
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
    name: String,
}

impl Name {
    fn new(name: &str) -> Self {
        Self { name: name.to_owned() }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename = "Level")]
pub struct LevelConfig<'a, T> {
    #[serde(borrow = "'a")]
    pub entities: Vec<EntityConfig<'a, T>>,
}

#[derive(thiserror::Error, Debug)]
pub enum LevelLoadError {
    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("RON parse error at {0}")]
    RonError(#[from] ron::de::SpannedError),
    #[error("Material error: {0}")]
    MaterialError(#[from] MaterialError),
    #[error("Mesh error: {0}")]
    MeshError(#[from] MeshError),
    #[error("Texture error: {0}")]
    TextureError(#[from] TextureError),
    #[error("No camera component found")]
    NoCameraComponent,
}

pub struct Level {
    pub world: World,
    camera: EntityId,
    pub meshes: Vec<Mesh>,
    pub material: Arc<Material>,
}

impl Level {
    pub fn new<T: DeserializableComponentConfig>(
        config_filepath: &Path,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> Result<Self, LevelLoadError> {
        // Parse config.
        let config_str = std::fs::read_to_string(config_filepath)?;
        let config = crate::utils::load_config::<LevelConfig<T>>(&config_str)?;

        let mut world = World::new();
        for entity_config in config.entities {
            let id = world.add_entity(Name::new(entity_config.name));
            for component_config in entity_config.components {
                component_config.load(id, &mut world, engine, cmd_pool)?;
            }
        }

        // Load texture.
        let _texture = engine.texture_manager.load_texture(
            "Viking room texture",
            &engine.game_dir.join("models/viking_room/viking_room.ktx2"),
            &engine.device,
            cmd_pool,
        )?;

        // Load material.
        let material = Arc::new(Material::new(
            &engine.pipeline_manager,
            &engine.game_dir.join("models/viking_room/viking_room.mat.ron"),
        )?);

        // Load mesh.
        let mesh = Mesh::new(
            "Viking room",
            &engine.device,
            cmd_pool,
            &engine.game_dir.join("models/viking_room/viking_room.obj"),
            Arc::clone(&material),
        )?;

        // Set up camera.
        let camera = world.run(|v_cameras: View<Camera>| -> Result<EntityId, LevelLoadError> {
            // Get the first entity with a camera component.
            let (id, _) = v_cameras
                .iter()
                .with_id()
                .next()
                .ok_or(LevelLoadError::NoCameraComponent)?;
            Ok(id)
        })?;
        //let camera_position = Vec3::new(2., 2., 2.);
        //let look_at = Mat4::look_at(camera_position, Vec3::zero(), Vec3::unit_z());
        //let camera_transform = Isometry3::new(look_at.extract_translation(), look_at.extract_rotation()).inversed();

        //let camera = Camera {
        //    transform: camera_transform,
        //    aspect: engine.get_window_aspect(),
        //    fov: 45.,
        //    near: 0.1,
        //};

        Ok(Self {
            world,
            camera,
            meshes: vec![mesh],
            material,
        })
    }

    pub fn get_camera(&mut self) -> (RefMut<&mut Isometry3>, Ref<&Camera>) {
        let (cam_transform, cam) = self
            .world
            .get::<(&mut IsometryComponent, &Camera)>(self.camera)
            .unwrap();
        (RefMut::map(cam_transform, |t| &mut t.0), cam)
    }
}
