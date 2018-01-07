use std::iter::Sum;
use std::ops::*;

use primitive::Primitive;

pub trait Float: 'static + Send + Sync + Copy + PartialEq + PartialOrd + Primitive
    + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self> + Sum
{
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn is_nan(self) -> bool;
    fn min(self, min: Self) -> Self;
    fn max(self, max: Self) -> Self;
}

impl Float for f32
{
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn sin_cos(self) -> (Self, Self) { self.sin_cos() }
    fn asin(self) -> Self { self.asin() }
    fn acos(self) -> Self { self.acos() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn min(self, min: Self) -> Self { self.min(min) }
    fn max(self, max: Self) -> Self { self.max(max) }
}

impl Float for f64
{
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn sin_cos(self) -> (Self, Self) { self.sin_cos() }
    fn asin(self) -> Self { self.asin() }
    fn acos(self) -> Self { self.acos() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn min(self, min: Self) -> Self { self.min(min) }
    fn max(self, max: Self) -> Self { self.max(max) }
}
