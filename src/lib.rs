#![cfg_attr(feature = "cargo-clippy", allow(many_single_char_names))]

#[cfg(test)]
#[macro_use]
mod test_helpers;

pub mod consts;
pub mod math;
pub mod matrix;
pub mod quaternion;
pub mod vector;

mod as_tuple;
mod float;
mod matrix_utilities;
mod primitive;

pub use matrix::{Mat2, Mat3, Mat4};
pub use quaternion::Quaternion;
pub use vector::{vec2, vec3, vec4, Vec2, Vec3, Vec4};
