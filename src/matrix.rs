use std::iter::Sum;
use std::mem::MaybeUninit;
use std::ops::*;

use crate::primitive::Primitive;
use crate::vector::*;

pub use crate::matrix_utilities::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayMat<T, const N: usize>(pub [[T; N]; N]);

impl<T: Copy + Default, const N: usize> Default for ArrayMat<T, N> {
    fn default() -> Self {
        ArrayMat([[T::default(); N]; N])
    }
}

impl<T, const N: usize> From<ArrayMat<T, N>> for [[T; N]; N]
where
    [[T; N]; N]: Clone,
{
    fn from(val: ArrayMat<T, N>) -> Self {
        val.0.clone()
    }
}

impl<T, const N: usize> ArrayMat<T, N> {
    pub fn new(array: [[T; N]; N]) -> Self {
        Self::from(array)
    }
}

impl<T, const N: usize> From<[[T; N]; N]> for ArrayMat<T, N> {
    fn from(array: [[T; N]; N]) -> Self {
        ArrayMat(array)
    }
}

impl<T: Copy, const N: usize> ArrayMat<T, N> {
    pub fn row(&self, index: usize) -> ArrayVec<T, N> {
        let mut result = unsafe { MaybeUninit::<[T; N]>::uninit().assume_init() };
        for i in 0..N {
            result[i] = self.0[i][index];
        }

        ArrayVec(result)
    }

    pub fn col(&self, index: usize) -> ArrayVec<T, N> {
        ArrayVec::new(self.0[index])
    }

    pub fn transpose(&self) -> Self {
        let mut result = unsafe { MaybeUninit::<[[T; N]; N]>::uninit().assume_init() };
        for i in 0..N {
            for j in 0..N {
                result[j][i] = self.0[i][j];
            }
        }
        ArrayMat(result)
    }
}

impl<T: Copy + Primitive, const N: usize> ArrayMat<T, N> {
    pub fn identity() -> Self {
        let mut result = [[T::zero(); N]; N];
        for i in 0..N {
            result[i][i] = T::one();
        }
        ArrayMat(result)
    }

    pub fn scale(scale: [T; N]) -> Self {
        let mut result = [[T::zero(); N]; N];
        for i in 0..N {
            result[i][i] = scale[i];
        }
        ArrayMat(result)
    }

    pub fn translation_homogenous(translation: [T; N]) -> Self {
        let mut result = Self::identity();
        result.0[N - 1] = translation;
        result
    }
}

impl<T: Copy, const N: usize> Mul for ArrayMat<T, N>
where
    T: Mul<Output = T> + Sum<T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let mut result = unsafe { MaybeUninit::<[[T; N]; N]>::uninit().assume_init() };
        for col in 0..N {
            for row in 0..N {
                result[col][row] = self.row(row).dot(other.col(col));
            }
        }
        ArrayMat(result)
    }
}

impl<T: Copy, const N: usize> Mul<ArrayVec<T, N>> for ArrayMat<T, N>
where
    T: Mul<Output = T> + Sum<T>,
{
    type Output = ArrayVec<T, N>;

    fn mul(self, vector: ArrayVec<T, N>) -> Self::Output {
        let mut result = unsafe { MaybeUninit::<[T; N]>::uninit().assume_init() };
        for i in 0..N {
            result[i] = self.row(i).dot(vector);
        }
        ArrayVec(result)
    }
}

pub type Mat1<T> = ArrayMat<T, 1>;
pub type Mat2<T> = ArrayMat<T, 2>;
pub type Mat3<T> = ArrayMat<T, 3>;
pub type Mat4<T> = ArrayMat<T, 4>;

fn extend_matrix<T: Copy + Primitive, const N: usize, const N_PLUS_1: usize>(
    mat: &ArrayMat<T, N>,
) -> ArrayMat<T, N_PLUS_1> {
    let mut result = ArrayMat::<T, N_PLUS_1>::identity();
    for i in 0..N {
        result.0[i][0..N].copy_from_slice(&mat.0[i]);
    }
    result
}

fn retract_matrix<T: Copy + Primitive, const N: usize, const N_MINUS_1: usize>(
    mat: &ArrayMat<T, N>,
) -> ArrayMat<T, N_MINUS_1> {
    let mut result = unsafe { MaybeUninit::<[[T; N_MINUS_1]; N_MINUS_1]>::uninit().assume_init() };
    for i in 0..N_MINUS_1 {
        result[i].copy_from_slice(&mat.0[i][0..N_MINUS_1]);
    }
    ArrayMat(result)
}

impl<T: Copy + Primitive> Mat1<T> {
    pub fn extend(&self) -> Mat2<T> {
        extend_matrix::<T, 1, 2>(self)
    }
}

impl<T: Copy + Primitive> Mat2<T> {
    pub fn extend(&self) -> Mat3<T> {
        extend_matrix::<T, 2, 3>(self)
    }

    pub fn retract(&self) -> Mat1<T> {
        retract_matrix::<T, 2, 1>(self)
    }

    pub fn translation(translation: [T; 1]) -> Self {
        Self::translation_homogenous(ArrayVec(translation).extend(T::one()).0)
    }
}

impl<T: Copy + Primitive> Mat3<T> {
    pub fn extend(&self) -> Mat4<T> {
        extend_matrix::<T, 3, 4>(self)
    }

    pub fn retract(&self) -> Mat2<T> {
        retract_matrix::<T, 3, 2>(self)
    }

    pub fn translation(translation: [T; 2]) -> Self {
        Self::translation_homogenous(ArrayVec(translation).extend(T::one()).0)
    }
}

impl<T: Copy + Primitive> Mat4<T> {
    pub fn retract(&self) -> Mat3<T> {
        retract_matrix::<T, 4, 3>(self)
    }

    pub fn translation(translation: [T; 3]) -> Self {
        Self::translation_homogenous(ArrayVec(translation).extend(T::one()).0)
    }
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for ArrayMat<T, N>
where
    [[T; N]; N]: Serialize,
{
    fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.0.serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Deserialize<'de> for ArrayMat<T, N>
where
    [[T; N]; N]: Deserialize<'de>,
{
    fn deserialize<D>(d: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(ArrayMat(<[[T; N]; N] as Deserialize<'de>>::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into() {
        let m2 = Mat2::new([[1, 2], [3, 4]]);
        let a: [[usize; 2]; 2] = m2.into();
        assert_eq!(a, [[1, 2], [3, 4]]);
    }

    #[test]
    fn identity() {
        let m2 = Mat2::new([[1, 0], [0, 1]]);
        let m3 = Mat3::new([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m4 = Mat4::new([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);

        assert_eq!(Mat2::identity(), m2);
        assert_eq!(Mat3::identity(), m3);
        assert_eq!(Mat4::identity(), m4);
    }

    #[test]
    fn transpose() {
        let m = Mat4::new([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
        let expected = Mat4::new([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]);
        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn rows_and_columns() {
        let a = vec4(0, 1, 2, 3);
        let b = vec4(4, 5, 6, 7);
        let c = vec4(8, 9, 10, 11);
        let d = vec4(12, 13, 14, 15);

        let m = Mat4::new([a.0, b.0, c.0, d.0]);
        let mt = m.transpose();

        assert_eq!(m.col(0), a);
        assert_eq!(m.col(1), b);
        assert_eq!(m.col(2), c);
        assert_eq!(m.col(3), d);

        assert_eq!(mt.row(0), a);
        assert_eq!(mt.row(1), b);
        assert_eq!(mt.row(2), c);
        assert_eq!(mt.row(3), d);
    }

    #[test]
    fn matrix_multiplication() {
        let a = Mat4::new([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]);
        let b = Mat4::new([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]]);
        let ab = Mat4::new([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [2, 4, 6, 1]]);
        let ba = Mat4::new([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [1, 2, 3, 1]]);
        assert_eq!(b * a, ba);
        assert_eq!(a * b, ab);
    }

    #[test]
    fn scaling() {
        let v = vec4(1, 2, 3, 4);
        let m = Mat4::scale([4, 3, 2, 1]);
        assert_eq!(m * v, vec4(4, 6, 6, 4));
    }

    #[test]
    fn translating() {
        let v = vec4(0, 0, 0, 1);
        let m = Mat4::translation([2, 4, 6]);
        assert_eq!(m * v, vec4(2, 4, 6, 1));
    }

    #[test]
    fn extending_and_retracting() {
        let m2 = Mat2::new([[1, 2], [5, 6]]);
        let m3 = Mat3::new([[1, 2, 3], [5, 6, 7], [9, 10, 11]]);
        let m4 = Mat4::new([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let e3 = Mat3::new([[1, 2, 0], [5, 6, 0], [0, 0, 1]]);
        let e4 = Mat4::new([[1, 2, 3, 0], [5, 6, 7, 0], [9, 10, 11, 0], [0, 0, 0, 1]]);

        assert_eq!(m4.retract(), m3);
        assert_eq!(m3.retract(), m2);
        assert_eq!(m2.extend(), e3);
        assert_eq!(m3.extend(), e4);
    }
}
