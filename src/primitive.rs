pub trait Primitive {
    fn zero() -> Self;
    fn one() -> Self;
    fn as_u8(&self) -> u8;
    fn as_u16(&self) -> u16;
    fn as_u32(&self) -> u32;
    fn as_u64(&self) -> u64;
    fn as_usize(&self) -> usize;
    fn as_i8(&self) -> i8;
    fn as_i16(&self) -> i16;
    fn as_i32(&self) -> i32;
    fn as_i64(&self) -> i64;
    fn as_isize(&self) -> isize;
    fn as_f32(&self) -> f32;
    fn as_f64(&self) -> f64;
}

macro_rules! impl_primitive {
    ($type_name: ty) => {
        #[allow(clippy::cast_lossless)]
        impl Primitive for $type_name {
            fn zero() -> Self {
                0 as $type_name
            }
            fn one() -> Self {
                1 as $type_name
            }
            fn as_u8(&self) -> u8 {
                *self as u8
            }
            fn as_u16(&self) -> u16 {
                *self as u16
            }
            fn as_u32(&self) -> u32 {
                *self as u32
            }
            fn as_u64(&self) -> u64 {
                *self as u64
            }
            fn as_usize(&self) -> usize {
                *self as usize
            }
            fn as_i8(&self) -> i8 {
                *self as i8
            }
            fn as_i16(&self) -> i16 {
                *self as i16
            }
            fn as_i32(&self) -> i32 {
                *self as i32
            }
            fn as_i64(&self) -> i64 {
                *self as i64
            }
            fn as_isize(&self) -> isize {
                *self as isize
            }
            fn as_f32(&self) -> f32 {
                *self as f32
            }
            fn as_f64(&self) -> f64 {
                *self as f64
            }
        }
    };
}

impl_primitive!(u8);
impl_primitive!(u16);
impl_primitive!(u32);
impl_primitive!(u64);
impl_primitive!(usize);
impl_primitive!(i8);
impl_primitive!(i16);
impl_primitive!(i32);
impl_primitive!(i64);
impl_primitive!(isize);
impl_primitive!(f32);
impl_primitive!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use std;

    #[test]
    fn float_zero_to_all() {
        assert_eq!((0.0).as_u8(), 0_u8);
        assert_eq!((0.0).as_u16(), 0_u16);
        assert_eq!((0.0).as_u32(), 0_u32);
        assert_eq!((0.0).as_u64(), 0_u64);
        assert_eq!((0.0).as_usize(), 0_usize);
        assert_eq!((0.0).as_i8(), 0_i8);
        assert_eq!((0.0).as_i16(), 0_i16);
        assert_eq!((0.0).as_i32(), 0_i32);
        assert_eq!((0.0).as_i64(), 0_i64);
        assert_eq!((0.0).as_isize(), 0_isize);
        assert_eq!((0.0).as_f32(), 0.0_f32);
        assert_eq!((0.0).as_f64(), 0.0_f64);
    }

    #[test]
    fn float_one_to_all() {
        assert_eq!((1.0).as_u8(), 1_u8);
        assert_eq!((1.0).as_u16(), 1_u16);
        assert_eq!((1.0).as_u32(), 1_u32);
        assert_eq!((1.0).as_u64(), 1_u64);
        assert_eq!((1.0).as_usize(), 1_usize);
        assert_eq!((1.0).as_i8(), 1_i8);
        assert_eq!((1.0).as_i16(), 1_i16);
        assert_eq!((1.0).as_i32(), 1_i32);
        assert_eq!((1.0).as_i64(), 1_i64);
        assert_eq!((1.0).as_isize(), 1_isize);
        assert_eq!((1.0).as_f32(), 1.0_f32);
        assert_eq!((1.0).as_f64(), 1.0_f64);
    }

    #[test]
    fn float_minus_one_to_all() {
        assert_eq!((-1.0).as_u8(), 0_u8);
        assert_eq!((-1.0).as_u16(), 0_u16);
        assert_eq!((-1.0).as_u32(), 0_u32);
        assert_eq!((-1.0).as_u64(), 0_u64);
        assert_eq!((-1.0).as_usize(), 0_usize);
        assert_eq!((-1.0).as_i8(), -1_i8);
        assert_eq!((-1.0).as_i16(), -1_i16);
        assert_eq!((-1.0).as_i32(), -1_i32);
        assert_eq!((-1.0).as_i64(), -1_i64);
        assert_eq!((-1.0).as_isize(), -1_isize);
        assert_eq!((-1.0).as_f32(), -1.0_f32);
        assert_eq!((-1.0).as_f64(), -1.0_f64);
    }

    #[test]
    fn float_to_all() {
        assert_eq!((8.0078125).as_u8(), 8_u8);
        assert_eq!((8.0078125).as_u16(), 8_u16);
        assert_eq!((8.0078125).as_u32(), 8_u32);
        assert_eq!((8.0078125).as_u64(), 8_u64);
        assert_eq!((8.0078125).as_usize(), 8_usize);
        assert_eq!((8.0078125).as_i8(), 8_i8);
        assert_eq!((8.0078125).as_i16(), 8_i16);
        assert_eq!((8.0078125).as_i32(), 8_i32);
        assert_eq!((8.0078125).as_i64(), 8_i64);
        assert_eq!((8.0078125).as_isize(), 8_isize);
        assert_eq!((8.0078125).as_f32(), 8.0078125_f32);
        assert_eq!((8.0078125).as_f64(), 8.0078125_f64);
    }

    #[test]
    fn int_zero_to_all() {
        assert_eq!((0).as_u8(), 0_u8);
        assert_eq!((0).as_u16(), 0_u16);
        assert_eq!((0).as_u32(), 0_u32);
        assert_eq!((0).as_u64(), 0_u64);
        assert_eq!((0).as_usize(), 0_usize);
        assert_eq!((0).as_i8(), 0_i8);
        assert_eq!((0).as_i16(), 0_i16);
        assert_eq!((0).as_i32(), 0_i32);
        assert_eq!((0).as_i64(), 0_i64);
        assert_eq!((0).as_isize(), 0_isize);
        assert_eq!((0).as_f32(), 0.0_f32);
        assert_eq!((0).as_f64(), 0.0_f64);
    }

    #[test]
    fn int_one_to_all() {
        assert_eq!((1).as_u8(), 1_u8);
        assert_eq!((1).as_u16(), 1_u16);
        assert_eq!((1).as_u32(), 1_u32);
        assert_eq!((1).as_u64(), 1_u64);
        assert_eq!((1).as_usize(), 1_usize);
        assert_eq!((1).as_i8(), 1_i8);
        assert_eq!((1).as_i16(), 1_i16);
        assert_eq!((1).as_i32(), 1_i32);
        assert_eq!((1).as_i64(), 1_i64);
        assert_eq!((1).as_isize(), 1_isize);
        assert_eq!((1).as_f32(), 1.0_f32);
        assert_eq!((1).as_f64(), 1.0_f64);
    }

    #[test]
    fn int_minus_one_to_all() {
        assert_eq!((-1).as_u8(), std::u8::MAX);
        assert_eq!((-1).as_u16(), std::u16::MAX);
        assert_eq!((-1).as_u32(), std::u32::MAX);
        assert_eq!((-1).as_u64(), std::u64::MAX);
        assert_eq!((-1).as_usize(), std::usize::MAX);
        assert_eq!((-1).as_i8(), -1_i8);
        assert_eq!((-1).as_i16(), -1_i16);
        assert_eq!((-1).as_i32(), -1_i32);
        assert_eq!((-1).as_i64(), -1_i64);
        assert_eq!((-1).as_isize(), -1_isize);
        assert_eq!((-1).as_f32(), -1.0_f32);
        assert_eq!((-1).as_f64(), -1.0_f64);
    }
}
