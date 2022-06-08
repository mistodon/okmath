use std::array::IntoIter;
use std::iter::{FromIterator, Sum};
use std::ops::*;

use crate::as_tuple::AsTuple;
use crate::float::Float;
use crate::helpers::collect_to_array;
use crate::primitive::Primitive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayVec<T, const N: usize>(pub [T; N]);

impl<T: Copy + Default, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        ArrayVec([T::default(); N])
    }
}

impl<T, const N: usize> Into<[T; N]> for ArrayVec<T, N>
where
    [T; N]: Clone
{
    fn into(self) -> [T; N] {
        self.0.clone()
    }
}

impl<T, const N: usize> ArrayVec<T, N> {
    pub const fn new(array: [T; N]) -> Self {
        ArrayVec(array)
    }
}

impl<T, const N: usize> From<[T; N]> for ArrayVec<T, N> {
    fn from(array: [T; N]) -> Self {
        ArrayVec(array)
    }
}

impl<T, const N: usize> FromIterator<T> for ArrayVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ArrayVec(collect_to_array(iter.into_iter()))
    }
}

impl<T: Copy, const N: usize> ArrayVec<T, N> {
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() >= N);
        Self::from_iter(slice.iter().copied())
    }

    #[inline(always)]
    pub fn map<F, U>(&self, operator: F) -> ArrayVec<U, N>
    where
        U: Copy,
        F: Fn(T) -> U,
    {
        ArrayVec(collect_to_array(IntoIter::new(self.0).map(operator)))
    }

    #[inline(always)]
    pub fn zipmap<F, U, V>(&self, other: ArrayVec<U, N>, operator: F) -> ArrayVec<V, N>
    where
        U: Copy,
        V: Copy,
        F: Fn(T, U) -> V,
    {
        ArrayVec(collect_to_array(
            IntoIter::new(self.0)
                .zip(IntoIter::new(other.0))
                .map(|(a, b)| operator(a, b)),
        ))
    }

    #[inline(always)]
    pub fn as_array(&self) -> [T; N] {
        self.0
    }
}

impl<T: Copy, const N: usize> ArrayVec<T, N>
where
    [T; N]: AsTuple,
{
    pub fn as_tuple(&self) -> <[T; N] as AsTuple>::Tuple {
        self.0.as_tuple()
    }
}

impl<T: Copy, const N: usize> ArrayVec<T, N>
where
    T: Primitive,
{
    pub fn as_u8(&self) -> ArrayVec<u8, N> {
        self.map(|x| x.as_u8())
    }
    pub fn as_u16(&self) -> ArrayVec<u16, N> {
        self.map(|x| x.as_u16())
    }
    pub fn as_u32(&self) -> ArrayVec<u32, N> {
        self.map(|x| x.as_u32())
    }
    pub fn as_u64(&self) -> ArrayVec<u64, N> {
        self.map(|x| x.as_u64())
    }
    pub fn as_usize(&self) -> ArrayVec<usize, N> {
        self.map(|x| x.as_usize())
    }
    pub fn as_i8(&self) -> ArrayVec<i8, N> {
        self.map(|x| x.as_i8())
    }
    pub fn as_i16(&self) -> ArrayVec<i16, N> {
        self.map(|x| x.as_i16())
    }
    pub fn as_i32(&self) -> ArrayVec<i32, N> {
        self.map(|x| x.as_i32())
    }
    pub fn as_i64(&self) -> ArrayVec<i64, N> {
        self.map(|x| x.as_i64())
    }
    pub fn as_isize(&self) -> ArrayVec<isize, N> {
        self.map(|x| x.as_isize())
    }
    pub fn as_f32(&self) -> ArrayVec<f32, N> {
        self.map(|x| x.as_f32())
    }
    pub fn as_f64(&self) -> ArrayVec<f64, N> {
        self.map(|x| x.as_f64())
    }
}

impl<T, const N: usize> Add for ArrayVec<T, N>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.zipmap(other, Add::add)
    }
}

impl<T, const N: usize> Sub for ArrayVec<T, N>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.zipmap(other, Sub::sub)
    }
}

impl<T, const N: usize> Mul for ArrayVec<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.zipmap(other, Mul::mul)
    }
}

impl<T, const N: usize> Div for ArrayVec<T, N>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        self.zipmap(other, Div::div)
    }
}

impl<T, const N: usize> Mul<T> for ArrayVec<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        self.map(|x| x * other)
    }
}

impl<T, const N: usize> Div<T> for ArrayVec<T, N>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        self.map(|x| x / other)
    }
}

impl<T: Copy, const N: usize> AddAssign for ArrayVec<T, N>
where
    Self: Add<Output = Self>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}

impl<T: Copy, const N: usize> SubAssign for ArrayVec<T, N>
where
    Self: Sub<Output = Self>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl<T: Copy, const N: usize> MulAssign for ArrayVec<T, N>
where
    Self: Mul<Output = Self>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other
    }
}

impl<T: Copy, const N: usize> DivAssign for ArrayVec<T, N>
where
    Self: Div<Output = Self>,
{
    fn div_assign(&mut self, other: Self) {
        *self = *self / other
    }
}

impl<T: Copy, const N: usize> MulAssign<T> for ArrayVec<T, N>
where
    Self: Mul<T, Output = Self>,
{
    fn mul_assign(&mut self, other: T) {
        *self = *self * other
    }
}

impl<T: Copy, const N: usize> DivAssign<T> for ArrayVec<T, N>
where
    Self: Div<T, Output = Self>,
{
    fn div_assign(&mut self, other: T) {
        *self = *self / other
    }
}

impl<T, const N: usize> Neg for ArrayVec<T, N>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        self.map(Neg::neg)
    }
}

impl<T, const N: usize> ArrayVec<T, N>
where
    Self: Mul<Output = Self>,
    T: Copy + Sum<T>,
{
    pub fn dot(&self, other: Self) -> T {
        (*self * other).0.iter().cloned().sum()
    }

    pub fn mag_sq(&self) -> T {
        self.dot(*self)
    }
}

impl<T, const N: usize> ArrayVec<T, N>
where
    Self: Mul<Output = Self> + Mul<T, Output = Self>,
    T: Copy + Sum<T> + Div<Output = T>,
{
    pub fn proj(&self, other: Self) -> Self {
        other * (self.dot(other) / other.dot(other))
    }
}

impl<T: Float, const N: usize> ArrayVec<T, N> {
    pub fn mag(&self) -> T {
        self.mag_sq().sqrt()
    }

    pub fn norm(&self) -> Self {
        let mag = self.mag();
        assert!(mag != T::zero(), "attempt to normalize zero vector");
        *self / mag
    }

    pub fn norm_zero(&self) -> Self {
        let mag = self.mag();
        if mag == T::zero() {
            *self
        } else {
            *self / mag
        }
    }
}

pub type Vec1<T> = ArrayVec<T, 1>;
pub type Vec2<T> = ArrayVec<T, 2>;
pub type Vec3<T> = ArrayVec<T, 3>;
pub type Vec4<T> = ArrayVec<T, 4>;

impl<T: Copy> Vec3<T>
where
    T: Mul<Output = T> + Sub<Output = T>,
{
    pub fn cross(&self, other: Self) -> Self {
        let (ax, ay, az) = self.as_tuple();
        let (bx, by, bz) = other.as_tuple();
        Vec3::new([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    }
}

impl<T: Copy> Vec1<T> {
    pub fn extend(&self, y: T) -> Vec2<T> {
        let (x,) = self.as_tuple();
        Vec2::new([x, y])
    }
}

impl<T: Copy> Vec2<T> {
    pub fn extend(&self, z: T) -> Vec3<T> {
        let (x, y) = self.as_tuple();
        Vec3::new([x, y, z])
    }
    pub fn retract(&self) -> Vec1<T> {
        let (x, _) = self.as_tuple();
        Vec1::new([x])
    }
}

impl<T: Copy> Vec3<T> {
    pub fn extend(&self, w: T) -> Vec4<T> {
        let (x, y, z) = self.as_tuple();
        Vec4::new([x, y, z, w])
    }
    pub fn retract(&self) -> Vec2<T> {
        let (x, y, _) = self.as_tuple();
        Vec2::new([x, y])
    }
}

impl<T: Copy> Vec4<T> {
    pub fn retract(&self) -> Vec3<T> {
        let (x, y, z, _) = self.as_tuple();
        Vec3::new([x, y, z])
    }
}

pub fn vec1<T: Copy>(x: T) -> Vec1<T> {
    Vec1::new([x])
}

pub fn vec2<T: Copy>(x: T, y: T) -> Vec2<T> {
    Vec2::new([x, y])
}

pub fn vec3<T: Copy>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3::new([x, y, z])
}

pub fn vec4<T: Copy>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4::new([x, y, z, w])
}

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde_support")]
impl<T, const N: usize> Serialize for ArrayVec<T, N>
where
    [T; N]: Serialize,
{
    fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.0.serialize(s)
    }
}

#[cfg(feature = "serde_support")]
impl<'de, T, const N: usize> Deserialize<'de> for ArrayVec<T, N>
where
    [T; N]: Deserialize<'de>,
{
    fn deserialize<D>(d: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(ArrayVec(<[T; N] as Deserialize<'de>>::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into() {
        let v2 = vec2(2, 3);
        let a: [usize; 2] = v2.into();
        assert_eq!(a, [2, 3]);
    }

    #[test]
    fn map() {
        let v2 = vec2(2, 3);
        let v3 = vec3(4, 5, 6);
        let v4 = vec4(7, 8, 9, 10);

        assert_eq!(v2.map(|x| x * x), vec2(4, 9));
        assert_eq!(v3.map(|x| x * x), vec3(16, 25, 36));
        assert_eq!(v4.map(|x| x * x), vec4(49, 64, 81, 100));
    }

    #[test]
    fn zipmap() {
        let a2 = vec2(2, 4);
        let b2 = vec2(3, 5);
        let a3 = vec3(1, 3, 5);
        let b3 = vec3(2, 4, 6);
        let a4 = vec4(2, 4, 6, 8);
        let b4 = vec4(1, 3, 5, 7);

        assert_eq!(a2.zipmap(b2, |x, y| x + y), vec2(5, 9));
        assert_eq!(a3.zipmap(b3, |x, y| x + y), vec3(3, 7, 11));
        assert_eq!(a4.zipmap(b4, |x, y| x + y), vec4(3, 7, 11, 15));
    }

    #[test]
    fn primitive_casts() {
        let f4 = vec4(1.0, 2.0, 3.0, 4.0);
        let i4 = vec4(1, 2, 3, 4);

        assert_eq!(f4.as_i32(), i4);
        assert_eq!(i4.as_f32(), f4);
    }

    #[test]
    fn as_tuple() {
        let (x, y, z, w) = vec4(0, 1, 2, 3).as_tuple();
        assert_eq!(x, 0);
        assert_eq!(y, 1);
        assert_eq!(z, 2);
        assert_eq!(w, 3);
    }

    #[test]
    fn as_array() {
        let a = [10, 20, 30, 40];
        let v = Vec4::new(a);
        assert_eq!(v.as_array(), a);
    }

    #[test]
    fn binary_operators() {
        let u = vec4(2, 6, 9, 12);
        let v = vec4(2, 3, 3, 2);
        assert_eq!(u + v, vec4(4, 9, 12, 14));
        assert_eq!(u - v, vec4(0, 3, 6, 10));
        assert_eq!(u * v, vec4(4, 18, 27, 24));
        assert_eq!(u / v, vec4(1, 2, 3, 6));
    }

    #[test]
    fn scalar_operators() {
        let u = vec4(2, 6, 8, 12);
        assert_eq!(u * 2, vec4(4, 12, 16, 24));
        assert_eq!(u / 2, vec4(1, 3, 4, 6));
    }

    #[test]
    fn in_place_binary_operators() {
        let v = vec4(2, 3, 3, 2);
        let mut a = vec4(2, 6, 9, 12);
        let mut b = vec4(2, 6, 9, 12);
        let mut c = vec4(2, 6, 9, 12);
        let mut d = vec4(2, 6, 9, 12);
        a += v;
        b -= v;
        c *= v;
        d /= v;
        assert_eq!(a, vec4(4, 9, 12, 14));
        assert_eq!(b, vec4(0, 3, 6, 10));
        assert_eq!(c, vec4(4, 18, 27, 24));
        assert_eq!(d, vec4(1, 2, 3, 6));
    }

    #[test]
    fn in_place_scalar_operators() {
        let mut u = vec4(2, 6, 8, 12);
        let mut v = vec4(2, 6, 8, 12);
        u *= 2;
        v /= 2;
        assert_eq!(u, vec4(4, 12, 16, 24));
        assert_eq!(v, vec4(1, 3, 4, 6));
    }

    #[test]
    fn negation() {
        let v = vec4(-1, 2, -3, 4);
        assert_eq!(-v, vec4(1, -2, 3, -4));
    }

    #[test]
    fn dot_product() {
        let a = vec4(1, 0, 0, 0);
        let b = vec4(0, 1, 0, 0);
        let c = vec4(-1, 0, 0, 0);
        let d = vec4(1, 1, 0, 0);
        assert_eq!(a.dot(a), 1);
        assert_eq!(a.dot(b), 0);
        assert_eq!(a.dot(c), -1);
        assert_eq!(d.dot(d), 2);
    }

    #[test]
    fn mag_sq() {
        let a = vec4(1, 2, 3, 4);
        assert_eq!(a.mag_sq(), 30);
    }

    #[test]
    fn magnitude() {
        let v = vec2(3.0, 4.0_f64);
        assert_eq!(v.mag(), 5.0);
    }

    #[test]
    fn projection() {
        let v = vec3(2, 4, 6);
        let onto = vec3(1, 2, 0);
        assert_eq!(v.proj(onto), vec3(2, 4, 0));
    }

    #[test]
    fn normalization() {
        let v = vec4(10.0, 0.0, 0.0, 0.0_f32);
        assert_eq!(v.norm(), vec4(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    #[should_panic]
    fn normalizing_zero_vector() {
        let z = vec4(0.0, 0.0, 0.0, 0.0_f32);
        let _y = z.norm();
    }

    #[test]
    fn cross_product() {
        let x = vec3(1, 0, 0);
        let y = vec3(0, 1, 0);
        let z = vec3(0, 0, 1);
        assert_eq!(x.cross(y), z);
        assert_eq!(y.cross(x), -z);
        assert_eq!(y.cross(z), x);
        assert_eq!(z.cross(y), -x);
        assert_eq!(z.cross(x), y);
        assert_eq!(x.cross(z), -y);
    }

    #[test]
    fn extending_vectors() {
        let u = vec3(1, 3, 5);
        let v = vec2(2, 4);

        assert_eq!(u.extend(7), vec4(1, 3, 5, 7));
        assert_eq!(v.extend(6), vec3(2, 4, 6));
    }

    #[test]
    fn retracting_vectors() {
        let u = vec3(1, 3, 5);
        let v = vec4(2, 4, 6, 8);

        assert_eq!(u.retract(), vec2(1, 3));
        assert_eq!(v.retract(), vec3(2, 4, 6));
    }

    #[test]
    fn from_slice_correct() {
        let s = &[1, 2, 3, 4, 5, 6];
        let v2 = Vec2::from_slice(s);
        let v3 = Vec3::from_slice(s);
        let v4 = Vec4::from_slice(s);
        assert_eq!(v2, vec2(1, 2));
        assert_eq!(v3, vec3(1, 2, 3));
        assert_eq!(v4, vec4(1, 2, 3, 4));
    }

    #[test]
    #[should_panic]
    fn from_slice_fail() {
        let s = &[1, 2, 3];
        let _v4 = Vec4::from_slice(s);
    }
}
