// Copyright (c) 2024 Ben Sutherland.

use std::slice;
use std::sync::Arc;

use arrayvec::ArrayVec;
use ash::vk;
use itertools::{izip, Itertools};
use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId, IntoIter, Ref, RefMut, View, ViewMut, World};

use crate::camera::{CamIsometry, Camera, FirstPersonCamera, ViewInfo};
use crate::engine::GalaxyEngine;
use crate::materials::{LoadingMaterialManager, Material, MaterialError, MaterialManager};
use crate::meshes::mesh_manager::{LoadingMeshManager, MeshManager};
use crate::meshes::{Mesh, MeshError};
use crate::pipelines::{GraphicsPipeline, Pipeline, PipelineManager};
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
        &mut self, // use self: Box<Self>, if going the boxed route with a custom deserialiser.
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
        &mut self,
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
        &mut self,
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
    scene_index: Option<u32>,
}

impl ComponentConfig for Transform {
    fn load(
        &mut self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        _engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.add_component(
            entity_id,
            TransformComponent {
                transform: self.clone(),
                scene_index: None,
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
        &mut self,
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
        &mut self,
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
        &mut self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        if self.materials.is_empty() {
            self.materials
                .push(LoadingMaterialManager::DEFAULT_MATERIAL.to_string());
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
                    material_path,
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
    pub material: vk::DeviceAddress,
    pub transform_index: u32,
    padding: u32,
    //pub material_index: u32,
}

pub struct LoadingLevel {
    config_path: ResourcePath,
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: LoadingMeshManager,
    pub material_manager: LoadingMaterialManager,
    pub texture_manager: TextureManager,
}

struct PipelineDrawSlice {
    pipeline: Arc<GraphicsPipeline>,
    offset: u32,
    len: u32,
}
impl PipelineDrawSlice {
    fn draw_offset_push_constant(&self) -> [u8; size_of::<u32>()] {
        self.offset.to_le_bytes()
    }

    fn draw_offset(&self) -> vk::DeviceSize {
        self.offset as vk::DeviceSize * size_of::<vk::DrawIndexedIndirectCommand>() as vk::DeviceSize
    }

    fn len(&self) -> u32 {
        self.len
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct SceneTransform {
    mvp: Mat4,
    inverse_transpose: Mat4,
}

pub struct Level {
    pub world: World,
    pub camera_entity: EntityId,
    pub mesh_manager: MeshManager,
    material_manager: MaterialManager,
    pub texture_manager: TextureManager,
    scene_descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    scene_uniform_buffer: VolatileBuffer<SceneUniformData>,
    scene_transforms_buffer: VolatileBuffer<SceneTransform>,
    draw_data_buffer: VolatileBuffer<DrawData>,
    draw_indirect_buffer: VolatileBuffer<crate::pod::vk::DrawIndexedIndirectCommand>,
    pipeline_draw_ranges: Vec<PipelineDrawSlice>,
}

impl Level {
    pub fn new<T: DeserializableComponentConfig>(
        config_path: ResourcePath,
        engine: &GalaxyEngine,
        cmd_pool: &mut TransientPrimaryCommandPool,
        _old_level: Option<Self>, // TODO: for reusing resources.
    ) -> Result<Self, LoadError> {
        // TODO: Unload previous level and resources.

        let mut texture_manager = TextureManager::new(&engine.device)?;
        let mut level = LoadingLevel {
            config_path,
            world: World::new(),
            camera_entity: EntityId::dead(),
            mesh_manager: LoadingMeshManager::new(),
            material_manager: LoadingMaterialManager::new(engine, &mut texture_manager, cmd_pool)?,
            texture_manager,
        };

        // Parse level config.
        let config_str = std::fs::read_to_string(level.config_path.full_path::<resource_type::Level>(engine))?;
        let config = crate::utils::load_ron_config::<LevelConfig<T>>(&config_str)?;

        // Load level.
        for entity_config in config.entities {
            let id = level.world.add_entity(Name::new(entity_config.name));
            for mut component_config in entity_config.components.into_iter() {
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
            // Transforms + draw data + material constants.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * 3),
            // Scene texture descriptor array.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(
                    (GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * level.texture_manager.num_textures()).max(1),
                ),
        ];
        let mut descriptor_pool = DescriptorPool::new(&engine.device, &descriptor_pool_sizes)?;

        descriptor_pool.allocate_descriptor_sets(
            &engine.device,
            &[engine.pipeline_manager.scene_set_layout; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        )?;

        // Create scene buffers.
        let scene_uniform_buffer = VolatileBuffer::new("Scene uniform buffer", device, VolatileBufferType::Uniform)?;

        // Count models and draw calls (one per mesh element).
        let (num_models, num_draws) = level
            .world
            .iter::<&Model>()
            .iter()
            .fold((0, 0), |(num_models, num_elements), model| {
                (num_models + 1, num_elements + model.mesh.num_elements() as usize)
            });

        // Transforms are per-model.
        let scene_transforms_buffer = VolatileBuffer::new_array(
            "Scene transforms buffer",
            num_models,
            device,
            VolatileBufferType::Storage,
        )?;

        // Finish material and mesh loading.
        let material_manager = level.material_manager.finalise_loading(engine, cmd_pool)?;
        let mesh_manager = level.mesh_manager.finalise_loading(engine, cmd_pool)?;

        let draw_data_buffer =
            VolatileBuffer::new_array("Draw data buffer", num_draws, device, VolatileBufferType::Storage)?;

        // Write to scene descriptor sets.
        let uniform_buffer_info = scene_uniform_buffer.descriptor_buffer_infos();
        let transform_buffer_info = scene_transforms_buffer.descriptor_buffer_infos();
        let draw_data_buffer_info = draw_data_buffer.descriptor_buffer_infos();
        let texture_image_infos = level.texture_manager.get_image_infos();
        //let material_buffer_info = material_manager.material_data_addresses_info();
        let material_constant_buffer_info = material_manager.material_constant_buffer_info();

        let mut descriptor_writes: ArrayVec<
            _,
            { GalaxyEngine::MAX_FRAMES_IN_FLIGHT * PipelineManager::NUM_SCENE_DESCRIPTOR_SET_BINDINGS },
        > = descriptor_pool
            .iter()
            .enumerate()
            .flat_map(
                |(frame, set)| -> [vk::WriteDescriptorSet; PipelineManager::NUM_SCENE_DESCRIPTOR_SET_BINDINGS - 1] {
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
                        // Draw data buffer.
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(slice::from_ref(&draw_data_buffer_info[frame])),
                        // Material buffers.
                        //vk::WriteDescriptorSet::default()
                        //    .dst_set(*set)
                        //    .dst_binding(3)
                        //    .dst_array_element(0)
                        //    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        //    .buffer_info(slice::from_ref(&material_buffer_info)),
                        // Material constants.
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(3)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(slice::from_ref(&material_constant_buffer_info)),
                    ]
                },
            )
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
            num_draws,
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
            draw_data_buffer,
            draw_indirect_buffer,
            pipeline_draw_ranges: Vec::new(),
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
            sun_direction: Vec3::new(time.sin() as f32, (time + 0.3).sin() as f32, (time + 0.6).sin() as f32)
                .normalized(),
            delta_time,
        };

        // Update scene data.
        self.world.run(
            |v_models: View<Model>, mut vm_transforms: ViewMut<TransformComponent>| {
                // Update scene transforms.
                {
                    let transform_buffer = self.scene_transforms_buffer.get_mut_slice(frame_index);

                    // Once models can be added and removed, the transforms buffer will need to be dynamically resized.
                    assert!(v_models.len() <= transform_buffer.len());

                    // Write transforms to buffer and save index. Only transforms for models are uploaded.
                    // TODO: Use wide for simd processing?
                    for (i, ((_model, transform_comp), transform_mat)) in (&v_models, &mut vm_transforms)
                        .iter()
                        .zip(transform_buffer.iter_mut())
                        .enumerate()
                    {
                        let transform = &transform_comp.transform;
                        transform_mat.mvp = view_info.mvp_from_transform(transform);
                        transform_mat.inverse_transpose = transform.to_inverse_transpose_matrix();

                        transform_comp.scene_index = Some(i as u32);
                    }
                }

                // Create array of all draw calls for the scene.
                // If transform indices become stable, the entries can be sorted on level load.
                struct SceneDrawEntry<'a> {
                    pipeline_id: &'a str,
                    mesh: &'a Mesh,
                    element_index: u32,
                    material: &'a Material,
                    transform_index: u32,
                }

                let mut scene_draws = (&v_models, &vm_transforms)
                    .iter()
                    .flat_map(|(model, transform_comp)| {
                        debug_assert_eq!(model.mesh.num_elements(), model.materials.len() as u32);
                        model.materials.iter().enumerate().map(|(i, mat)| SceneDrawEntry {
                            pipeline_id: mat.pipeline().id(),
                            mesh: &model.mesh,
                            element_index: i as u32,
                            material: mat,
                            transform_index: transform_comp.scene_index.unwrap(),
                        })
                    })
                    .collect::<Vec<_>>();

                // Sort by pipeline id, then mesh, then mesh element.
                scene_draws.sort_by(|a, b| {
                    a.pipeline_id
                        .cmp(b.pipeline_id)
                        .then(a.mesh.level_index().cmp(&b.mesh.level_index()))
                        .then(a.element_index.cmp(&b.element_index))
                });

                // Construct the ranges of draws using the same pipeline.
                self.pipeline_draw_ranges = scene_draws
                    .iter()
                    .enumerate()
                    .peekable()
                    .batching(|scene_draws| {
                        let (start, start_draw) = scene_draws.peek()?;
                        let offset = *start as u32;
                        let pipeline = start_draw.material.cloned_pipeline();

                        let len = scene_draws
                            .clone()
                            .take_while(|(_, sd)| sd.pipeline_id == pipeline.id())
                            .count();
                        scene_draws.nth(len - 1);
                        Some(PipelineDrawSlice {
                            pipeline,
                            offset,
                            len: len as u32,
                        })
                    })
                    .collect();

                // Write draw indirect commands.
                for (draw_data, draw_indirect, scene_draw) in izip!(
                    self.draw_data_buffer.get_mut_slice(frame_index).iter_mut(),
                    self.draw_indirect_buffer.get_mut_slice(frame_index).iter_mut(),
                    scene_draws.iter()
                ) {
                    draw_data.transform_index = scene_draw.transform_index;
                    draw_data.material = self.material_manager.get_material_addr(scene_draw.material);

                    let draw_params = self.mesh_manager.get_element_draw_data_for_mesh(scene_draw.mesh)
                        [scene_draw.element_index as usize];
                    draw_indirect.index_count = draw_params.index_count;
                    draw_indirect.instance_count = 1;
                    draw_indirect.first_index = draw_params.first_index;
                    draw_indirect.vertex_offset = draw_params.vertex_offset;
                    draw_indirect.first_instance = 0;
                }
            },
        );
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

        // Megabuffer has all the mesh vertex and index data for the scene.
        self.mesh_manager.bind_megabuffer(cmd_buf);

        for pipeline_draw_range in self.pipeline_draw_ranges.iter() {
            let pipeline = pipeline_draw_range.pipeline.as_ref();
            assert_eq!(pipeline.layout(), layout);

            // Bind pipeline.
            cmd_buf.bind_graphics_pipeline(pipeline);
            cmd_buf.push_constants(
                layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &pipeline_draw_range.draw_offset_push_constant(),
            );
            unsafe {
                device.loader().cmd_draw_indexed_indirect(
                    cmd_buf.handle_dep(),
                    self.draw_indirect_buffer.handle_dep(),
                    self.draw_indirect_buffer.frame_offset(frame_index) as vk::DeviceSize
                        + pipeline_draw_range.draw_offset(),
                    pipeline_draw_range.len(),
                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                )
            };
        }
    }
}
