// Copyright (c) 2024 Ben Sutherland.

use serde::{Deserialize, Serialize};
use shipyard::{Component, EntityId};

use crate::engine::GalaxyEngine;
use crate::level::{ComponentConfig, LoadResult, LoadingLevel};
use crate::prelude::*;
use crate::vulkan::command_buffer::TransientPrimaryCommandPool;

#[derive(Serialize, Deserialize, Debug)]
pub struct CameraConfig {
    fov: f32,
    near: f32,
}

impl ComponentConfig for CameraConfig {
    fn load(
        &mut self,
        entity_id: EntityId,
        level: &mut LoadingLevel,
        engine: &GalaxyEngine,
        _cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> LoadResult<()> {
        level.world.add_component(
            entity_id,
            Camera {
                aspect: engine.get_window_aspect(),
                fov: self.fov,
                near: self.near,
            },
        );
        level.camera_entity = entity_id;
        Ok(())
    }
}

// Define the camera-specific coordinate system.
pub trait CamIsometry {
    fn cam_up(&self) -> Vec3;
    fn cam_right(&self) -> Vec3;
    fn cam_forward(&self) -> Vec3;
}

impl CamIsometry for Isometry3 {
    fn cam_up(&self) -> Vec3 {
        self.rotation * Vec3::unit_y()
    }
    fn cam_right(&self) -> Vec3 {
        self.rotation * Vec3::unit_x()
    }
    // The camera forward vector is the negative z-axis in view space.
    // Because the engine uses right-handed coordinates, the camera view space (y-up) is also
    // right-handed, which means the positive z-axis points opposite to the view direction.
    fn cam_forward(&self) -> Vec3 {
        self.rotation * -Vec3::unit_z()
    }
}

#[derive(Component, Debug)]
pub struct Camera {
    pub aspect: f32,
    pub fov: f32, // degrees
    pub near: f32,
}

pub trait FirstPersonCamera {
    fn apply_first_person_mouse(&mut self, mouse: Vec2);
}

impl FirstPersonCamera for Isometry3 {
    fn apply_first_person_mouse(&mut self, mouse: Vec2) {
        self.rotation = Rotor3::from_angle_plane(mouse.x.to_radians(), Bivec3::from_normalized_axis(Vec3::unit_z()))
            * self.rotation;
        self.rotation = Rotor3::from_angle_plane(mouse.y.to_radians(), Bivec3::from_normalized_axis(self.cam_right()))
            * self.rotation;
    }
}

// Calculates the camera view and projection transforms for rendering.
pub struct ViewInfo {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
}

impl ViewInfo {
    pub fn new(camera: &Camera, transform: &Isometry3) -> Self {
        let view = transform.inversed().into_homogeneous_matrix();
        let projection = ultraviolet::projection::perspective_reversed_infinite_z_vk(
            camera.fov.to_radians(),
            camera.aspect,
            camera.near,
        );

        Self {
            view,
            projection,
            view_projection: projection * view,
        }
    }

    pub fn mvp_from_matrix(&self, mat: Mat4) -> Mat4 {
        self.view_projection * mat
    }

    pub fn mvp_from_similarity(&self, sim: &Similarity3) -> Mat4 {
        self.mvp_from_matrix(sim.into_homogeneous_matrix())
    }

    pub fn mvp_from_transform(&self, transform: &Transform) -> Mat4 {
        self.mvp_from_matrix(transform.to_matrix())
    }
}
