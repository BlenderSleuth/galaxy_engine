use nalgebra as na;
use nalgebra::{Perspective3, RealField};

pub type Vec2 = na::Vector2<f32>;
pub type Vec3 = na::Vector3<f32>;
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