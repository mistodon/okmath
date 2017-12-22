use std::iter::Sum;
use std::ops::*;

use primitive::Primitive;
use vector::*;


pub use matrix_utilities::*;


macro_rules! matrix_type
{
    ($name: ident, $vec: ident, $smaller_vec: ident, $size: tt) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name<T: Copy>(pub [[T; $size]; $size]);

        impl<T: Copy> $name<T>
        {
            pub fn row(&self, index: usize) -> $vec<T>
            {
                let mut result: $vec<T> = unsafe { ::std::mem::uninitialized() };

                for i in 0..$size
                {
                    result.0[i] = self.0[i][index];
                }

                result
            }

            pub fn col(&self, index: usize) -> $vec<T>
            {
                $vec(self.0[index])
            }

            pub fn transpose(&self) -> Self
            {
                let mut result: Self = unsafe { ::std::mem::uninitialized() };

                for col in 0..$size
                {
                    for row in 0..$size
                    {
                        result.0[col][row] = self.0[row][col];
                    }
                }

                result
            }
        }

        impl<T> $name<T>
        where
            T: Copy + Primitive
        {
            pub fn identity() -> Self
            {
                let mut result: Self = unsafe { ::std::mem::uninitialized() };

                for col in 0..$size
                {
                    for row in 0..$size
                    {
                        result.0[col][row] = if col == row { T::one() } else { T::zero() };
                    }
                }

                result
            }

            pub fn scale(scale: $vec<T>) -> Self
            {
                let mut base = Self::identity();
                for i in 0..$size
                {
                    base.0[i][i] = scale.0[i];
                }
                base
            }

            pub fn translation(translation: $smaller_vec<T>) -> Self
            {
                let mut base = Self::identity();
                base.0[$size - 1] = translation.extend(T::one()).0;
                base
            }
        }

        impl<T: Copy> Mul for $name<T>
        where
            T: Mul<Output=T> + Sum<T>
        {
            type Output = Self;

            fn mul(self, other: Self) -> Self::Output
            {
                let mut result: Self = unsafe { ::std::mem::uninitialized() };

                for col in 0..$size
                {
                    for row in 0..$size
                    {
                        result.0[col][row] = self.row(row).dot(other.col(col));
                    }
                }

                result
            }
        }

        impl<T: Copy> Mul<$vec<T>> for $name<T>
        where
            T: Mul<Output=T> + Sum<T>
        {
            type Output = $vec<T>;

            fn mul(self, vector: $vec<T>) -> Self::Output
            {
                let mut result: Self::Output = unsafe { ::std::mem::uninitialized() };

                for i in 0..$size
                {
                    result.0[i] = self.row(i).dot(vector);
                }

                result
            }
        }
    }
}


matrix_type!(Mat2, Vec2, Vec1, 2);
matrix_type!(Mat3, Vec3, Vec2, 3);
matrix_type!(Mat4, Vec4, Vec3, 4);


#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn identity()
    {
        let m = Mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        assert_eq!(Mat4::identity(), m);
    }

    #[test]
    fn transpose()
    {
        let m = Mat4([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]);
        let expected = Mat4([
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15]
        ]);
        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn rows_and_columns()
    {
        let a = vec4(0, 1, 2, 3);
        let b = vec4(4, 5, 6, 7);
        let c = vec4(8, 9, 10, 11);
        let d = vec4(12, 13, 14, 15);

        let m = Mat4([a.0, b.0, c.0, d.0]);
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
    fn matrix_multiplication()
    {
        let a = Mat4([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]);
        let b = Mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 1]]);
        let ab = Mat4([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [2, 4, 6, 1]]);
        let ba = Mat4([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [1, 2, 3, 1]]);
        assert_eq!(b * a, ba);
        assert_eq!(a * b, ab);
    }

    #[test]
    fn scaling()
    {
        let v = vec4(1, 2, 3, 4);
        let m = Mat4::scale(vec4(4, 3, 2, 1));
        assert_eq!(m * v, vec4(4, 6, 6, 4));
    }

    #[test]
    fn translating()
    {
        let v = vec4(0, 0, 0, 1);
        let m = Mat4::translation(vec3(2, 4, 6));
        assert_eq!(m * v, vec4(2, 4, 6, 1));
    }
}

