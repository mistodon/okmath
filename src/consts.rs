pub const TAU32: f32 = ::std::f32::consts::PI * 2.0;
pub const TAU64: f64 = ::std::f64::consts::PI * 2.0;


#[cfg(test)]
mod tests
{
    use consts::*;
    use std::f64::EPSILON;

    #[test]
    fn trigonometric_functions()
    {
        assert!(TAU64.sin() - 0.0 < EPSILON);
        assert!(TAU64.cos() - 1.0 < EPSILON);

        let (s, c) = TAU64.sin_cos();
        assert!(s - 0.0 < EPSILON);
        assert!(c - 1.0 < EPSILON);
    }
}
