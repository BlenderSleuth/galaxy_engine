// Copyright (c) 2024 Ben Sutherland.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::utils::ConfigID;

#[typetag::serde(tag = "type")]
pub trait ComponentConfig: Debug {}

#[derive(Serialize, Deserialize, Debug)]
struct Vector3(f32, f32, f32);
#[derive(Serialize, Deserialize, Debug)]
struct Rotor(f32, f32, f32, f32);

#[derive(Serialize, Deserialize, Debug)]
struct Transform {
    pub position: Vector3,
    pub rotation: Rotor,
    pub scale: Vector3,
}

#[typetag::serde]
impl ComponentConfig for Transform {}

#[derive(Serialize, Deserialize, Debug)]
struct Camera {}

#[typetag::serde]
impl ComponentConfig for Camera {}

#[derive(Serialize, Deserialize, Debug)]
struct Light {
    pub colour: Vector3,
    pub intensity: f32,
}

#[typetag::serde]
impl ComponentConfig for Light {}

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    pub mesh: ConfigID,
    pub material: ConfigID,
}

#[typetag::serde]
impl ComponentConfig for Model {}

#[derive(Deserialize, Debug)]
pub struct Entity {
    pub name: ConfigID,
    pub components: Vec<Box<dyn ComponentConfig>>,
}

#[derive(Deserialize, Debug)]
pub struct Scene {
    pub entities: Vec<Entity>,
}
