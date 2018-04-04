use std::iter::Sum;
use std::ops::*;

use primitive::Primitive;
use vector::*;


pub use matrix_utilities::*;


macro_rules! matrix_type
{
    (
        $name: ident,
        $vec: ident,
        $smaller_vec: ident,
        $size: tt,
        [$($index: tt),*],
        { $($col: tt : [$($row: tt),*]),* },
        [$([$($id: ident),*]),*]) =>
    {
        #[cfg_attr(feature = "serde_support", derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize))]
        #[cfg_attr(not(feature = "serde_support"), derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash))]
        pub struct $name<T: Copy>(pub [[T; $size]; $size]);

        impl<T: Copy> $name<T>
        {
            pub fn row(&self, index: usize) -> $vec<T>
            {
                $vec([
                    $(
                        self.0[$index][index]
                    ),*
                ])
            }

            pub fn col(&self, index: usize) -> $vec<T>
            {
                $vec(self.0[index])
            }

            pub fn transpose(&self) -> Self
            {
                $name([
                    $(
                        [ $(self.0[$row][$col]),* ]
                    ),*
                ])
            }
        }

        impl<T> $name<T>
        where
            T: Copy + Primitive
        {
            pub fn identity() -> Self
            {
                $name([
                    $(
                        [ $(T::$id()),* ]
                    ),*
                ])
            }

            // TODO(***realname***): Report bug to clippy? This can't be a memcpy.
            #[cfg_attr(feature = "cargo-clippy", allow(manual_memcpy))]
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
                $name([
                    $(
                        [ $( self.row($row).dot(other.col($col))),* ]
                    ),*
                ])
            }
        }

        impl<T: Copy> Mul<$vec<T>> for $name<T>
        where
            T: Mul<Output=T> + Sum<T>
        {
            type Output = $vec<T>;

            fn mul(self, vector: $vec<T>) -> Self::Output
            {
                $vec([
                    $(
                        self.row($index).dot(vector)
                    ),*
                ])
            }
        }
    }
}


matrix_type!(Mat2, Vec2, Vec1, 2, [0, 1],
    {
        0: [0, 1],
        1: [0, 1]
    },
    [
        [one, zero],
        [zero, one]
    ]
);

matrix_type!(Mat3, Vec3, Vec2, 3, [0, 1, 2],
    {
        0: [0, 1, 2],
        1: [0, 1, 2],
        2: [0, 1, 2]
    },
    [
        [one, zero, zero],
        [zero, one, zero],
        [zero, zero, one]
    ]
);

matrix_type!(Mat4, Vec4, Vec3, 4, [0, 1, 2, 3],
    {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3],
        3: [0, 1, 2, 3]
    },
    [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, one]
    ]
);

impl<T> Mat2<T>
where
    T: Copy + Primitive
{
    pub fn extend(&self) -> Mat3<T>
    {
        let mut m = Mat3::identity();
        for col in 0..2
        {
            m.0[col][..2].clone_from_slice(&self.0[col][..2]);
        }
        m
    }
}

impl<T> Mat3<T>
where
    T: Copy + Primitive
{
    pub fn extend(&self) -> Mat4<T>
    {
        let mut m = Mat4::identity();
        for col in 0..3
        {
            m.0[col][..3].clone_from_slice(&self.0[col][..3]);
        }
        m
    }

    pub fn retract(&self) -> Mat2<T>
    {
        let mut m = Mat2::identity();
        for col in 0..2
        {
            m.0[col][..2].clone_from_slice(&self.0[col][..2]);
        }
        m
    }
}

impl<T> Mat4<T>
where
    T: Copy + Primitive
{
    pub fn retract(&self) -> Mat3<T>
    {
        let mut m = Mat3::identity();
        for col in 0..3
        {
            m.0[col][..3].clone_from_slice(&self.0[col][..3]);
        }
        m
    }
}


#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn identity()
    {
        let m2 = Mat2([[1, 0], [0, 1]]);
        let m3 = Mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let m4 = Mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);

        assert_eq!(Mat2::identity(), m2);
        assert_eq!(Mat3::identity(), m3);
        assert_eq!(Mat4::identity(), m4);
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

    #[test]
    fn extending_and_retracting()
    {
        let m2 = Mat2([[1, 2], [5, 6]]);
        let m3 = Mat3([[1, 2, 3], [5, 6, 7], [9, 10, 11]]);
        let m4 = Mat4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let e3 = Mat3([[1, 2, 0], [5, 6, 0], [0, 0, 1]]);
        let e4 = Mat4([[1, 2, 3, 0], [5, 6, 7, 0], [9, 10, 11, 0], [0, 0, 0, 1]]);

        assert_eq!(m4.retract(), m3);
        assert_eq!(m3.retract(), m2);
        assert_eq!(m2.extend(), e3);
        assert_eq!(m3.extend(), e4);
    }
}

