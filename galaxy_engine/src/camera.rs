// Copyright (c) 2024. Ben Sutherland

use crate::prelude::*;

pub struct Camera {
    pub transform: Isometry3,
    pub aspect: f32,
    pub fov: f32, // degrees
    pub near: f32,
}

impl Camera {
    pub fn up(&self) -> Vec3 {
        self.transform.rotation * Vec3::unit_y()
    }
    pub fn down(&self) -> Vec3 {
        -self.up()
    }
    pub fn right(&self) -> Vec3 {
        self.transform.rotation * Vec3::unit_x()
    }
    pub fn left(&self) -> Vec3 {
        -self.right()
    }

    pub fn backward(&self) -> Vec3 {
        self.transform.rotation * Vec3::unit_z()
    }

    // The direction the camera is facing.
    // Because the engine uses right-handed coordinates, the camera view space (y-up) is also
    // right-handed, which means the positive z-axis points opposite to the view direction.
    pub fn forward(&self) -> Vec3 {
        -self.backward()
    }

    pub fn view_info(&self) -> ViewInfo {
        ViewInfo::new(self)
    }
}

pub trait FirstPersonCamera {
    fn apply_first_person_mouse(&mut self, mouse: Vec2);
}

impl FirstPersonCamera for Camera {
    fn apply_first_person_mouse(&mut self, mouse: Vec2) {
        self.transform.rotation =
            Rotor3::from_angle_plane(mouse.x.to_radians(), Bivec3::from_normalized_axis(Vec3::unit_z()))
                * self.transform.rotation;
        self.transform.rotation =
            Rotor3::from_angle_plane(mouse.y.to_radians(), Bivec3::from_normalized_axis(self.right()))
                * self.transform.rotation;
    }
}

// Calculates the camera view and projection transforms for rendering.
pub struct ViewInfo {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_projection: Mat4,
}

impl ViewInfo {
    pub fn new(camera: &Camera) -> Self {
        let view = camera.transform.inversed().into_homogeneous_matrix();
        let projection =
            ultraviolet::projection::perspective_infinite_z_vk(camera.fov.to_radians(), camera.aspect, camera.near);

        Self {
            view,
            projection,
            view_projection: projection * view,
        }
    }

    pub fn mvp_from_similarity(&self, sim: &Similarity3) -> Mat4 {
        sim.into_homogeneous_matrix() * self.view_projection
    }
}
