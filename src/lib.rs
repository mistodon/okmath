#![cfg_attr(feature = "cargo-clippy", allow(many_single_char_names))]

#[cfg(feature = "serde_support")]
#[macro_use]
extern crate serde_derive;

#[cfg(feature = "serde_support")]
extern crate serde;

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
mod primitive;
mod matrix_utilities;


pub use matrix::{ Mat2, Mat3, Mat4 };
pub use quaternion::{ Quaternion };
pub use vector::{ Vec2, Vec3, Vec4, vec2, vec3, vec4 };
