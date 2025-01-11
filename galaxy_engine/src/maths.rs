// Copyright (c) 2024-2025 Ben Sutherland.

use serde::{Deserialize, Serialize};
pub use ultraviolet::{Bivec3, Isometry3, Mat3, Mat4, Rotor3, Similarity3, Vec2, Vec3, Vec4};

use crate::vulkan::physical_device::PhysicalDevice;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Rotor3,
    pub scale: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::zero(),
            rotation: Rotor3::identity(),
            scale: 1.,
        }
    }
}

impl Transform {
    pub fn invertible(&self) -> bool {
        //self.scale.x.abs() > f32::EPSILON && self.scale.y.abs() > f32::EPSILON && self.scale.z.abs() > f32::EPSILON
        self.scale.abs() > f32::EPSILON
    }

    pub fn to_matrix(&self) -> Mat4 {
        (self.rotation.into_matrix() * Mat3::from_scale(self.scale))
            .into_homogeneous()
            .translated(&self.translation)
    }

    pub fn inverse(&self) -> Self {
        // Ensure invertible.
        assert!(self.invertible());

        let inv_rotation = self.rotation.reversed();
        //let inv_scale = Vec3::new(1. / self.scale.x, 1. / self.scale.y, 1. / self.scale.z);
        let inv_scale = self.scale.recip();
        let mut inv_translation = self.translation * inv_scale;
        inv_rotation.rotate_vec(&mut inv_translation);
        inv_translation = -inv_translation;

        Self {
            translation: inv_translation,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }

    //pub fn to_inverse_transpose_matrix(&self) -> Mat3 {
    //    // Ensure invertible.
    //    assert!(self.invertible());

    //    let inv_rotation = self.rotation.reversed();
    //    let inv_scale = Vec3::new(1. / self.scale.x, 1. / self.scale.y, 1. / self.scale.z);
    //    (Mat3::from_nonuniform_scale(inv_scale) * inv_rotation.into_matrix()).transposed()
    //}

    //// Rotor component of the inverse transpose matrix.
    //pub fn to_inverse_transpose_rotor(&self) -> Rotor3 {
    //    self.to_inverse_transpose_matrix().into_rotor3()
    //}
}

pub fn rotor_to_shader_quat(rotor: Rotor3) -> [f32; 4] {
    [rotor.bv.xz, rotor.bv.yz, rotor.bv.xy, rotor.s]
}

pub fn to_unorm(v: f32) -> u8 {
    (v * 255.).round() as u8
}

// Euler angles in degrees.
// Applied in the order yaw, pitch, roll.
#[derive(Debug, Clone, Copy)]
pub struct EulerAngles {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

pub trait Mat3Ext {
    fn to_euler_angles(&self) -> EulerAngles;
    fn from_euler_angles_new(angles: EulerAngles) -> Self;
}

impl Mat3Ext for Mat3 {
    fn to_euler_angles(&self) -> EulerAngles {
        let x = self.cols[0];
        let y = self.cols[1];
        let z = self.cols[2];
        let mut result = EulerAngles {
            pitch: x.z.atan2((x.x * x.x + x.y * x.y).sqrt()).to_degrees(),
            yaw: x.y.atan2(x.x).to_degrees(),
            roll: 0.,
        };
        let result_mat =
            Mat3::from_rotation_z(result.yaw.to_radians()) * Mat3::from_rotation_y(result.pitch.to_radians());
        let final_y_axis = result_mat * Vec3::unit_y();
        result.roll = z.dot(final_y_axis).atan2(y.dot(final_y_axis)).to_degrees();

        result
    }

    fn from_euler_angles_new(angles: EulerAngles) -> Self {
        let mut result = Mat3::identity();
        let (sp, cp) = angles.pitch.to_radians().sin_cos();
        let (sy, cy) = angles.yaw.to_radians().sin_cos();
        let (sr, cr) = angles.roll.to_radians().sin_cos();

        result[0][0] = cp * cy;
        result[0][1] = cp * sy;
        result[0][2] = sp;

        result[1][0] = sr * sp * cy - cr * sy;
        result[1][1] = sr * sp * sy + cr * cy;
        result[1][2] = -sr * cp;

        result[2][0] = -(cr * sp * cy + sr * sy);
        result[2][1] = cy * sr - cr * sp * sy;
        result[2][2] = cr * cp;

        result
    }
}

pub fn spin_transform(time_s: f32, rpm: f32) -> Similarity3 {
    Similarity3::new(
        Vec3::zero(),
        Rotor3::from_angle_plane(
            time_s * 360_f32.to_radians() * rpm / 60.,
            Bivec3::from_normalized_axis(Vec3::unit_z()),
        ),
        0.,
    )
}

// Finds a 2D grid size for a given count of elements, with a maximum number of unused elements.
pub fn grid_size_for_count(count: u32, max_unused: u32, max_side_length: u32) -> Option<(u32, u32)> {
    if count <= max_side_length {
        return Some((count, 1));
    }
    if count >= max_side_length * max_side_length {
        return None;
    }
    let max_unused = max_unused as i32;
    let mut root = (count as f32).sqrt().floor() as u32;
    while {
        let diff = (root * count.div_ceil(root)) as i32 - count as i32;
        0 > diff || diff > max_unused
    } {
        root -= 1;
    }
    Some((root, count.div_ceil(root)))
}

#[allow(dead_code)]
pub(crate) fn test_grid_size_for_count(test_size: u32) {
    let mut results = Vec::with_capacity(test_size as usize);
    for test_num in 1..test_size {
        let (group_count_x, group_size_y) =
            grid_size_for_count(test_num, 5, PhysicalDevice::MAX_DISPATCH_GROUPS_PER_DIMENSION).unwrap();
        let grid_size = group_count_x * group_size_y;
        let remainder = grid_size as i32 - test_num as i32;
        results.push((test_num, group_count_x, group_size_y, remainder));
    }
    for (test_num, x, y, remainder) in results {
        println!("Test num: {test_num}, {x}x{y}, remainder: {remainder}",);
    }
}
