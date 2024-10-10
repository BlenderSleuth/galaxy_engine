use ash::vk;
use nalgebra as na;
use nalgebra::{Isometry3, Perspective3, RealField, Rotation3};

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
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl ModelViewProjection {
    pub fn spin(window_size: vk::Extent2D, time: f32, rpm: f32) -> Self {
        Self {
            model: Rotation3::from_axis_angle(&na::UnitVector3::new_normalize(na::Vector3::new(0., 0., 1.)), time * 360_f32.to_radians() * rpm / 60.).to_homogeneous(),
            view: Isometry3::look_at_rh(&na::Point3::new(2., 2., 2.), &na::Point3::new(0., 0., 0.), &na::Vector3::new(0., 0., 1.)).to_homogeneous(),
            proj: Perspective3::vk_new(window_size.width as f32 / window_size.height as f32, 45_f32.to_radians(), 0.1, 10.0).to_homogeneous(),
        }
    }

    pub fn mvp(&self) -> Mat4 {
        self.proj * self.view * self.model
    }
}

impl Default for ModelViewProjection {
    fn default() -> Self {
        Self {
            model: Mat4::identity(),
            view: Mat4::identity(),
            proj: Mat4::identity(),
        }
    }
}
