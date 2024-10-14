// Copyright (c) 2024. Ben Sutherland

use ash::vk;
use nalgebra as na;
use nalgebra::{Affine3, Isometry3, Perspective3, RealField, Rotation3};

// pub type Vec2 = na::Vector2<f32>;
// pub type Vec3 = na::Vector3<f32>;
// pub type Vec4 = na::Vector4<f32>;
pub type Mat4 = na::Matrix4<f32>;

// Creates a new perspective matrix in Vulkan's coordinate system.
pub trait VkPerspective<T: RealField> {
    fn vk_new(aspect: T, fovy: T, znear: T, zfar: T) -> Self;
}

impl<T: RealField> VkPerspective<T> for Perspective3<T> {
    fn vk_new(aspect: T, fovy: T, znear: T, zfar: T) -> Self {
        let mut result = Perspective3::new(aspect, fovy, znear, zfar).into_inner();
        // Flip the y-axis to match Vulkan's coordinate system.
        result[(1, 1)] *= -T::one();
        Perspective3::from_matrix_unchecked(result)
    }
}

pub struct ModelViewProjection {
    model: Affine3<f32>,
    view: Isometry3<f32>,
    proj: Perspective3<f32>,
}

impl ModelViewProjection {
    pub fn spin(window_size: vk::Extent2D, time_s: f32, rpm: f32) -> Self {
        Self {
            model: na::convert(Rotation3::from_axis_angle(
                &na::UnitVector3::new_normalize(na::Vector3::new(0., 0., 1.)),
                time_s * 360_f32.to_radians() * rpm / 60.,
            )),
            view: Isometry3::look_at_rh(
                &na::Point3::new(2., 2., 2.),
                &na::Point3::new(0., 0., 0.),
                &na::Vector3::new(0., 0., 1.),
            ),
            proj: Perspective3::vk_new(
                window_size.width as f32 / window_size.height as f32,
                45_f32.to_radians(),
                0.1,
                10.0,
            ),
        }
    }

    pub fn mvp(&self) -> Mat4 {
        self.proj.as_matrix() * (self.view * self.model).to_homogeneous()
    }

    pub fn push_constant_range() -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<Mat4>() as u32)
    }
}

impl Default for ModelViewProjection {
    fn default() -> Self {
        Self {
            model: Affine3::identity(),
            view: Isometry3::identity(),
            proj: Perspective3::new(1., 1., 0., 1.),
        }
    }
}
