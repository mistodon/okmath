use std::iter::Sum;
use std::ops::*;

use float::Float;
use primitive::Primitive;
use matrix::{ Mat3, Mat4 };
use vector::{ Vec3, vec3 };


#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Quaternion<T: Copy>(pub T, pub Vec3<T>);


impl<T: Copy + Primitive> Quaternion<T>
{
    pub fn identity() -> Self
    {
        Quaternion(T::one(), vec3(T::zero(), T::zero(), T::zero()))
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Neg<Output=T>
{
    pub fn conj(&self) -> Self
    {
        Quaternion(self.0, -self.1)
    }
}

impl<T> Add for Quaternion<T>
where
    T: Copy + Add<Output=T>
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output
    {
        Quaternion(self.0 + other.0, self.1 + other.1)
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Sum<T>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output
    {
        let real = self.0 * other.0 - self.1.dot(other.1);
        let imvec = other.1 * self.0 + self.1 * other.0 + self.1.cross(other.1);
        Quaternion(real, imvec)
    }
}

impl<T> Mul<Vec3<T>> for Quaternion<T>
where
    T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Sum<T> + Primitive + Neg<Output=T>
{
    type Output = Vec3<T>;

    fn mul(self, vector: Vec3<T>) -> Self::Output
    {
        let v = Quaternion(T::zero(), vector);
        (self * v * self.conj()).1
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Copy + Mul<Output=T>
{
    type Output = Self;

    fn mul(self, constant: T) -> Self::Output
    {
        Quaternion(self.0 * constant, self.1 * constant)
    }
}

impl<T> Quaternion<T>
where
    T: Copy + Add<Output=T> + Mul<Output=T> + Sum<T>
{
    pub fn dot(&self, other: Self) -> T { self.0 * other.0 + self.1.dot(other.1) }

    pub fn mag_sq(&self) -> T { self.dot(*self) }
}

impl<T: Float + ::std::fmt::Debug> Quaternion<T>
{
    pub fn mag(&self) -> T { self.mag_sq().sqrt() }

    pub fn norm(&self) -> Self
    {
        let mag = self.mag();
        assert!(mag != T::zero(), "attempt to normalize zero quaternion");
        Quaternion(self.0 / mag, self.1 / mag)
    }

    pub fn axis_angle(axis: Vec3<T>, angle: T) -> Self
    {
        let a = angle / (T::one() + T::one());
        let (s, c) = a.sin_cos();
        Quaternion(c, axis * s)
    }

    pub fn euler_angles(x: T, y: T, z: T) -> Self
    {
        let two = T::one() + T::one();
        let (sx, cx) = (x / two).sin_cos();
        let (sy, cy) = (y / two).sin_cos();
        let (sz, cz) = (z / two).sin_cos();

        Quaternion(
            cx * cy * cz + sx * sy * sz,
            vec3(
                sx * cy * cz + cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz + sx * sy * cz,
            ))
    }

    pub fn slerp(&self, other: Self, t: T) -> Self
    {
        let it = T::one() - t;
        let mags = self.mag() * other.mag();
        assert!(mags != T::zero());

        let dot_mags = self.dot(other) / mags;
        let a = dot_mags.min(T::one()).max(-T::one()).acos();
        let sina = a.sin();
        if sina == T::zero() { *self } else { *self * ((it*a).sin() / sina) + other * ((t*a).sin() / sina) }
    }
}

impl<T> From<Quaternion<T>> for Mat3<T>
where
    T: Copy + Primitive + Add<Output=T> + Sub<Output=T> + Mul<Output=T>
{
    fn from(q: Quaternion<T>) -> Self
    {
        let (x, y, z) = q.1.as_tuple();
        let w = q.0;
        let one = T::one();
        let two = one + one;

        Mat3([
            [one - two*y*y - two*z*z, two*x*y + two*w*z, two*x*z - two*w*y],
            [two*x*y - two*w*z, one - two*x*x - two*z*z, two*y*z + two*w*x],
            [two*x*z + two*w*y, two*y*z - two*w*x, one - two*x*x - two*y*y],
        ])
    }
}


impl<T> From<Quaternion<T>> for Mat4<T>
where
    T: Copy + Primitive + Add<Output=T> + Sub<Output=T> + Mul<Output=T>
{
    fn from(q: Quaternion<T>) -> Self
    {
        Mat3::from(q).extend()
    }
}


#[cfg(test)]
mod tests
{
    use super::*;
    use consts::TAU32;

    fn quat<T: Copy>(w: T, x: T, y: T, z: T) -> Quaternion<T>
    {
        Quaternion(w, vec3(x, y, z))
    }

    #[test]
    fn dot_product()
    {
        let qw = quat(1, 0, 0, 0);
        let qx = quat(0, 1, 0, 0);
        let qy = quat(0, 0, 1, 0);
        let qz = quat(0, 0, 0, 1);
        let qa = quat(1, 2, 3, 4);
        let qb = quat(2, 3, 4, 5);

        assert_eq!(qw.dot(qx), 0);
        assert_eq!(qx.dot(qy), 0);
        assert_eq!(qy.dot(qz), 0);
        assert_eq!(qz.dot(qw), 0);
        assert_eq!(qw.dot(qw), 1);
        assert_eq!(qx.dot(qx), 1);
        assert_eq!(qa.dot(qb), 40);
    }

    #[test]
    fn mag_sq()
    {
        let qa = quat(1, 2, 3, 4);
        assert_eq!(qa.mag_sq(), 30);
    }

    #[test]
    fn mag()
    {
        let q = quat(3.0, 0.0, 0.0, 0.0_f32);
        assert_eq!(q.mag(), 3.0);
    }

    #[test]
    fn norm()
    {
        let q = quat(4.0, 0.0, 0.0, 0.0_f32);
        assert_eq!(q.norm(), quat(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn zero_rotation_is_identity()
    {
        let id = Quaternion::identity();
        let x = Quaternion::axis_angle(vec3(1.0, 0.0, 0.0), 0.0_f32);
        let y = Quaternion::axis_angle(vec3(0.0, 1.0, 0.0), 0.0_f32);
        let z = Quaternion::axis_angle(vec3(0.0, 0.0, 1.0), 0.0_f32);
        assert_eq!(x, id);
        assert_eq!(y, id);
        assert_eq!(z, id);
    }

    #[test]
    fn quaternion_multiplication()
    {
        let a4 = TAU32;
        let a = a4 / 4.0;
        let a2 = a4 / 2.0;
        let q = Quaternion::axis_angle(vec3(1.0, 0.0, 0.0), a);
        let q2 = Quaternion::axis_angle(vec3(1.0, 0.0, 0.0), a2);
        let q4 = Quaternion::axis_angle(vec3(1.0, 0.0, 0.0), a4);
        assert_quat_eq!(q*q, q2);
        assert_quat_eq!(q2*q2, q4);
        assert_quat_eq!(q*q*q*q, q4);
    }

    #[test]
    fn quaternion_vector_multiplication()
    {
        let v = vec3(1.0, 0.0, 0.0);
        let q = Quaternion::axis_angle(vec3(0.0, 0.0, 1.0), TAU32 / 4.0);
        assert_vec_eq!(q * v, vec3(0.0, 1.0, 0.0));

        let q4 = Quaternion::axis_angle(vec3(0.0, 0.0, 1.0), TAU32);
        assert_vec_eq!(q4*v, v);
    }

    #[test]
    fn quaternion_to_matrix()
    {
        use matrix;

        let q = Quaternion::axis_angle(vec3(0.0, 1.0, 0.0), TAU32 / 4.0);
        let m = matrix::axis_rotation(vec3(0.0, 1.0, 0.0), TAU32 / 4.0);
        let qm = Mat4::from(q);

        assert_mat_eq!(m, qm);
    }

    #[test]
    fn euler_rotation()
    {
        let qx = Quaternion::euler_angles(TAU32 / 4.0, 0.0, 0.0);
        let qy = Quaternion::euler_angles(0.0, TAU32 / 4.0, 0.0);
        let qz = Quaternion::euler_angles(0.0, 0.0, TAU32 / 4.0);
        let vx = vec3(1.0, 0.0, 0.0);
        let vy = vec3(0.0, 1.0, 0.0);
        let vz = vec3(0.0, 0.0, 1.0);
        assert_vec_eq!(qx * vy, vz);
        assert_vec_eq!(qy * vz, vx);
        assert_vec_eq!(qz * vx, vy);
    }

    #[test]
    fn slerp()
    {
        let q0 = Quaternion::euler_angles(0.0, 0.0, 0.0);
        let q1 = Quaternion::euler_angles(TAU32 / 4.0, 0.0, 0.0);
        let q = Quaternion::euler_angles(TAU32 / 8.0, 0.0, 0.0);
        assert_quat_eq!(q0.slerp(q1, 0.0), q0);
        assert_quat_eq!(q0.slerp(q1, 0.5), q);
        assert_quat_eq!(q0.slerp(q1, 1.0), q1);
    }

    #[test]
    fn slerp_to_self()
    {
        let q0 = Quaternion::identity();
        assert_quat_eq!(q0.slerp(q0, 0.0), q0);
        assert_quat_eq!(q0.slerp(q0, 0.5), q0);
        assert_quat_eq!(q0.slerp(q0, 1.0), q0);
    }
}
