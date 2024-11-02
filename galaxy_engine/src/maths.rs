// Copyright (c) 2024. Ben Sutherland

pub use ultraviolet::{Bivec3, Isometry3, Mat3, Mat4, Rotor3, Similarity3, Vec2, Vec3, Vec4};

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
