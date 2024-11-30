// Copyright (c) 2024 Ben Sutherland.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use ultraviolet::Vec3;

//#[derive(Serialize, Deserialize, Debug)]
//pub struct Vector3(f32, f32, f32);
//
//impl From<Vector3> for ultraviolet::Vec3 {
//    fn from(value: Vector3) -> Self {
//        Self::new(value.0, value.1, value.2)
//    }
//}
//
//#[derive(Serialize, Deserialize, Debug)]
//pub struct Rotor(f32, f32, f32, f32);
//
//impl From<Rotor> for ultraviolet::Rotor3 {
//    fn from(value: Rotor) -> Self {
//        Self::new(value.0, ultraviolet::Bivec3::new(value.1, value.2, value.3))
//    }
//}
