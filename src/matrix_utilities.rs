use crate::matrix::Mat4;
use crate::vector::*;

pub fn ortho_projection(aspect: f32, size: f32, near: f32, far: f32) -> Mat4<f32> {
    let inv_width = 1.0 / (aspect * size);
    let inv_height = 1.0 / size;

    Mat4::new([
        [inv_width, 0.0, 0.0, 0.0],
        [0.0, inv_height, 0.0, 0.0],
        [0.0, 0.0, 2.0 / (far - near), 0.0],
        [0.0, 0.0, -(far + near) / (far - near), 1.0],
    ])
}

// TODO: Make this work with standard 0-1 depth range
pub fn perspective_projection(aspect: f32, fov: f32, near: f32, far: f32) -> Mat4<f32> {
    let f = 1.0 / (fov / 2.0).tan();
    let f_a = f / aspect;

    Mat4::new([
        [f_a, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (far - near), 1.0],
        [0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0],
    ])
}

// TODO: Update for 0-1 depth
pub fn invert_perspective_matrix(matrix: Mat4<f32>) -> Mat4<f32> {
    let a = matrix.0[0][0];
    let b = matrix.0[1][1];
    let c = matrix.0[2][2];
    let _c1 = matrix.0[2][3];
    let d = matrix.0[3][2];

    debug_assert!(_c1 == 1.);

    Mat4::from([
        [1. / a, 0., 0., 0.],
        [0., 1. / b, 0., 0.],
        [0., 0., 0., 1. / d],
        [0., 0., 1., -c / d],
    ])
}

pub fn invert_view_matrix(matrix: Mat4<f32>) -> Mat4<f32> {
    // º
    // https://stackoverflow.com/questions/155670/invert-4x4-matrix-numerical-most-stable-solution-needed/155705#155705
    let u = Vec4::from(matrix.0[0]);
    let v = Vec4::from(matrix.0[1]);
    let w = Vec4::from(matrix.0[2]);
    let t = Vec4::from(matrix.0[3]);

    Mat4::from([
        [u.0[0], v.0[0], w.0[0], 0.],
        [u.0[1], v.0[1], w.0[1], 0.],
        [u.0[2], v.0[2], w.0[2], 0.],
        [-(u.dot(t)), -(v.dot(t)), -(w.dot(t)), 1.],
    ])
}

pub fn axis_rotation(axis: [f32; 3], angle: f32) -> Mat4<f32> {
    let (x, y, z) = Vec3::new(axis).norm_zero().as_tuple();
    let (s, c) = angle.sin_cos();
    let ic = 1.0 - c;

    Mat4::new([
        [
            (c + x * x * ic),
            (y * x * ic + z * s),
            (z * x * ic - y * s),
            0.0,
        ],
        [
            (x * y * ic - z * s),
            (c + y * y * ic),
            (z * y * ic + x * s),
            0.0,
        ],
        [
            (x * z * ic + y * s),
            (y * z * ic - x * s),
            (c + z * z * ic),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn euler_rotation(angles: [f32; 3]) -> Mat4<f32> {
    let (x, y, z) = Vec3::new(angles).as_tuple();
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();

    Mat4::new([
        [cz * cy, sz * cy, -sy, 0.0],
        [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx, 0.0],
        [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

pub fn look_rotation(forward: [f32; 3], up: [f32; 3]) -> Mat4<f32> {
    let forward = Vec3::new(forward).norm_zero();
    let up = Vec3::new(up).norm_zero();
    let right = up.cross(forward).norm_zero();
    let up = forward.cross(right);

    Mat4::new([
        right.extend(0.0).0,
        up.extend(0.0).0,
        forward.extend(0.0).0,
        [0.0, 0.0, 0.0, 1.0],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::TAU32;

    /// Rounds a number, but only if it's almost an integer
    fn fudge(x: f32) -> f32 {
        let f = x.fract().abs();
        if f < 0.00001 || f > 0.99999 {
            x.round()
        } else {
            panic!("Weird number: {x} (fractional part is: {f})");
        }
    }

    #[test]
    fn orthographic_projection() {
        let m = ortho_projection(1.0, 4.0, 0.0, 4.0);
        let p = vec4(4.0, 4.0, 4.0, 1.0);
        assert_eq!(m * p, vec4(1.0, 1.0, 1.0, 1.0));
    }

    // TODO: Wrong for Vulkan depth - see implementation
    #[test]
    fn perspective_projection_transformation() {
        let m = perspective_projection(1.0, TAU32 / 4.0, 1.0, 2.0);
        let u = vec4(1.0, 1.0, 1.0, 1.0);
        let v = vec4(2.0, 2.0, 2.0, 1.0);

        assert_eq!(m * u, vec4(1.0, 1.0, -1.0, 1.0));
        assert_eq!(m * v, vec4(2.0, 2.0, 2.0, 2.0));
    }

    #[test]
    fn perspective_matrix_inversion() {
        let m = perspective_projection(1.0, TAU32 / 4.0, 1.0, 2.0);
        let u_view = vec4(1.0, 1.0, 1.0, 1.0);
        let v_view = vec4(2.0, 2.0, 2.0, 1.0);

        let m_i = invert_perspective_matrix(m);
        let u_screen = vec4(1.0, 1.0, -1.0, 1.0);
        let v_screen = vec4(2.0, 2.0, 2.0, 2.0);

        assert_eq!(m_i * m, Mat4::identity());
        assert_eq!(m_i * u_screen, u_view);
        assert_eq!(m_i * v_screen, v_view);
    }

    #[test]
    fn simple_view_matrix_inversion() {
        // Move 1 forward, then rotate 90° to face X
        let m = euler_rotation([0., TAU32 / 4., 0.]) * Mat4::translation([0., 0., 1.]);

        let u = vec4(0., 0., 1., 1.);
        let expected = vec4(2., 0., 0., 1.);

        assert_eq!((m * u).map(fudge), expected);

        let m_i = invert_view_matrix(m);
        assert_eq!(m_i * m, Mat4::identity());
        assert_eq!((m_i * expected).map(fudge), u);
    }

    #[test]
    fn complicated_view_matrix_inversion() {
        // Move 1 forward, then rotate 90° to face X, then move 1, then rotate to -Z
        let m = euler_rotation([0., TAU32 / 4., 0.])
            * Mat4::translation([1., 0., 0.])
            * euler_rotation([0., TAU32 / 4., 0.])
            * Mat4::translation([0., 0., 1.]);

        let u = vec4(0., 0., 1., 1.);
        let expected = vec4(0., 0., -3., 1.);

        assert_eq!((m * u).map(fudge), expected);

        let m_i = invert_view_matrix(m);
        assert_eq!(m_i * m, Mat4::identity());
        assert_eq!((m_i * expected).map(fudge), u);
    }

    // TODO: 0-1 depth again lol
    #[test]
    fn point_to_ray_test_for_my_own_sanity() {
        // Scenario is: camera looking right, 45° FOV, one unit back
        //  near plane at zero, far plane at 1.0.
        //  Object at <1, 0, 2> world space should be hit by a
        //  "ray" cast from the middle-right side of the viewport.
        //  (because it is on the 4x4 unit far plane, 2 from the centre along Z)

        let world_space_point = vec4(1., 0., 2., 1.);
        let view_space_point = vec4(-2., 0., 2., 1.);
        let screen_space_point = vec4(-2., 0., 2., 2.);

        // Camera facing +X, one unit back from origin
        let view_matrix = euler_rotation([0., -TAU32 / 4., 0.]) * Mat4::translation([1., 0., 0.]);

        let test_points = [
            // Camera position -> origin
            (vec4(-1., 0., 0., 1.), vec4(0., 0., 0., 1.)),
            // Origin is 1 unit away from camera
            (vec4(0., 0., 0., 1.), vec4(0., 0., 1., 1.)),
            // Camera faces along X, so x=1 -> z=2
            (vec4(1., 0., 0., 1.), vec4(0., 0., 2., 1.)),
            (world_space_point, view_space_point),
        ];

        for (before, after) in test_points {
            assert_eq!((view_matrix * before).map(fudge), after);
        }

        let near = 1.;
        let far = 2.;
        let projection_matrix = perspective_projection(1., TAU32 / 4., near, far);

        let test_points = [
            (vec4(0., 0., 1., 1.), vec4(0., 0., -1., 1.)),
            (vec4(1., 1., 1., 1.), vec4(1., 1., -1., 1.)),
            (vec4(0., 0., 2., 1.), vec4(0., 0., 2., 2.)),
            (vec4(2., -2., 2., 1.), vec4(2., -2., 2., 2.)),
            (view_space_point, screen_space_point),
        ];

        for (before, after) in test_points {
            assert_eq!((projection_matrix * before).map(fudge), after);
        }

        let inv_proj = invert_perspective_matrix(projection_matrix);
        let inv_view = invert_view_matrix(view_matrix);

        // Using -1. as near plane, bc my matrices are bad
        let screen_space_point = vec4(-1., 0., -1., 1.);
        let unprojected_point = inv_proj * screen_space_point;
        assert_eq!(unprojected_point.map(fudge), vec4(-1., 0., 1., 1.));

        let near_plane_world_point = inv_view * unprojected_point;
        assert_eq!(near_plane_world_point.map(fudge), vec4(0., 0., 1., 1.));

        let camera_position = inv_view * vec4(0., 0., 0., 1.);
        assert_eq!(camera_position.map(fudge), vec4(-1., 0., 0., 1.));

        let ro = camera_position;
        let rd = (near_plane_world_point - camera_position).norm();
        let t = (2_f32).sqrt() * 2.;
        assert_eq!((ro + rd * t).map(fudge), world_space_point);
    }

    #[test]
    fn rotation_around_axis() {
        let m = axis_rotation([0.0, 1.0, 0.0], ::std::f32::consts::PI / 2.0);
        let x = vec4(1.0, 0.0, 0.0, 1.0);
        let y = vec4(0.0, 1.0, 0.0, 1.0);
        let z = vec4(0.0, 0.0, 1.0, 1.0);
        let mx = (m * x).map(fudge);
        let my = (m * y).map(fudge);
        let mz = (m * z).map(fudge);
        assert_eq!(mx, vec4(0.0, 0.0, -1.0, 1.0));
        assert_eq!(my, vec4(0.0, 1.0, 0.0, 1.0));
        assert_eq!(mz, vec4(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn euler_rotation_of_vector() {
        let a = ::std::f32::consts::PI / 2.0;

        let u = vec4(1.0, 0.0, 0.0, 1.0);

        let mx = euler_rotation([a, 0.0, 0.0]);
        let my = euler_rotation([0.0, a, 0.0]);
        let mz = euler_rotation([0.0, 0.0, a]);

        let ux = (mx * u).map(fudge);
        let uy = (my * u).map(fudge);
        let uz = (mz * u).map(fudge);

        assert_eq!(ux, vec4(1.0, 0.0, 0.0, 1.0));
        assert_eq!(uy, vec4(0.0, 0.0, -1.0, 1.0));
        assert_eq!(uz, vec4(0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn look_at_rotation() {
        let m = look_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        let x = vec4(1.0, 0.0, 0.0, 1.0);
        let y = vec4(0.0, 1.0, 0.0, 1.0);
        let z = vec4(0.0, 0.0, 1.0, 1.0);

        let mx = (m * x).map(fudge);
        let my = (m * y).map(fudge);
        let mz = (m * z).map(fudge);

        assert_eq!(mx, vec4(0.0, 0.0, -1.0, 1.0));
        assert_eq!(my, vec4(0.0, 1.0, 0.0, 1.0));
        assert_eq!(mz, vec4(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn look_rotation_correctly_normalized() {
        let m = look_rotation([0.0, -1.0, 2.0], [0.0, 1.0, 0.0]);

        let x = vec4(1.0, 0.0, 0.0, 1.0);
        let y = vec4(0.0, 1.0, 0.0, 1.0);
        let z = vec4(0.0, 0.0, 1.0, 1.0);

        let dx = (m * x).retract().mag();
        let dy = (m * y).retract().mag();
        let dz = (m * z).retract().mag();

        println!("{} {} {}", dx, dy, dz);

        assert!((1.0 - dx).abs() < 0.01);
        assert!((1.0 - dy).abs() < 0.01);
        assert!((1.0 - dz).abs() < 0.01);
    }
}
