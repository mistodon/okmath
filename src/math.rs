use std::ops::*;

pub fn lerp<C, Time>(from: C, to: C, t: Time) -> C
where
    C: Add<Output = C> + Sub<Output = C> + Mul<Time, Output = C> + Copy,
{
    from + (to - from) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::*;

    #[test]
    fn lerp_floats() {
        assert_eq!(lerp(0.0, 2.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 2.0, 0.5), 1.0);
        assert_eq!(lerp(0.0, 2.0, 1.0), 2.0);
        assert_eq!(lerp(10.0, 20.0, 0.5), 15.0);
    }

    #[test]
    fn lerp_vectors() {
        let u = vec4(1.0, 2.0, 3.0, 4.0);
        let v = vec4(5.0, 6.0, 7.0, 8.0);

        assert_eq!(lerp(u, v, 0.0), u);
        assert_eq!(lerp(u, v, 0.5), vec4(3.0, 4.0, 5.0, 6.0));
        assert_eq!(lerp(u, v, 1.0), v);
    }
}
