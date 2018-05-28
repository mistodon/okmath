extern crate okmath;

use okmath::*;


#[test]
fn basic_vector_usage()
{
    let a = vec3(1, 2, 3);
    let b = vec3(4, 5, 6);

    let (x, y, z) = a.as_tuple();
    let c = vec3(z, y, x);

    assert_eq!(a + b, vec3(5, 7, 9));
    assert_eq!(b - a, vec3(3, 3, 3));
    assert_eq!(a * b, vec3(4, 10, 18));
    assert_eq!(c, vec3(3, 2, 1));
}


#[test]
fn complex_vector_operations()
{
    let a = vec3(1, 2, 3);
    assert_eq!(a.dot(a), 14);

    let b = vec2(3.0, 4.0_f32);
    assert_eq!(b.mag(), 5.0);
}


#[test]
fn casting()
{
    let high_p = vec4(1.0, 2.0, 3.0, 4.0_f64);
    let low_p = vec4(1.0, 2.0, 3.0, 4.0_f32);
    assert_eq!(high_p.as_f32(), low_p);
}


#[test]
fn matrix_transformations()
{
    let angle = ::std::f32::consts::PI / 2.0;
    let a = Mat4::identity();
    let b = Mat4::scale([2.0, 3.0, 4.0, 1.0]);
    let c = Mat4::translation([1.0, 1.0, 1.0]);
    let d = matrix::euler_rotation([0.0, angle, 0.0]);

    let m = d * c * b * a;

    let v = vec4(1.0, 1.0, 1.0, 1.0);
    let mv = (m * v).map(f32::round);

    assert_eq!(mv, vec4(5.0, 4.0, -3.0, 1.0));
}

