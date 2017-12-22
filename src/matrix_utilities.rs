use matrix::{ Mat4 };
use vector::*;


pub fn ortho_projection(aspect: f32, size: f32, near: f32, far: f32) -> Mat4<f32>
{
    let inv_width = 1.0 / (aspect * size);
    let inv_height = 1.0 / size;

    Mat4([
         [inv_width, 0.0, 0.0, 0.0],
         [0.0, inv_height, 0.0, 0.0],
         [0.0, 0.0, 2.0 / (far - near), 0.0],
         [0.0, 0.0, -(far + near) / (far - near), 1.0]
    ])
}


pub fn axis_rotation(axis: [f32; 3], angle: f32) -> Mat4<f32>
{
    let (x, y, z) = Vec3(axis).norm().as_tuple();
    let (s, c) = angle.sin_cos();
    let ic = 1.0 - c;

    Mat4([
         [(c + x * x * ic), (y * x * ic + z * s), (z * x * ic - y * s), 0.0],
         [(x * y * ic - z * s), (c + y * y * ic), (z * y * ic + x * s), 0.0],
         [(x * z * ic + y * s), (y * z * ic - x * s), (c + z * z * ic), 0.0],
         [0.0, 0.0, 0.0, 1.0],
    ])

}


pub fn euler_rotation(angles: [f32; 3]) -> Mat4<f32>
{
    let (x, y, z) = Vec3(angles).as_tuple();
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();

    Mat4([
         [cz * cy, sz * cy, -sy, 0.0],
         [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx, 0.0],
         [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx, 0.0],
         [0.0, 0.0, 0.0, 1.0]
    ])
}


pub fn look_at(forward: [f32; 3], up: [f32; 3]) -> Mat4<f32>
{
    let forward = Vec3(forward).norm();
    let up = Vec3(up).norm();
    let right = up.cross(forward);
    let up = forward.cross(right);

    fn zero_w(v: Vec3<f32>) -> [f32; 4]
    {
        let (x, y, z) = v.as_tuple();
        [x, y, z, 0.0]
    }

    Mat4([zero_w(right), zero_w(up), zero_w(forward), [0.0, 0.0, 0.0, 1.0]])
}


#[cfg(test)]
mod tests
{
    #[ignore]
    #[test]
    fn orthographic_projection()
    {
        panic!()
    }

    #[ignore]
    #[test]
    fn axis_rotation()
    {
        panic!()
    }

    #[ignore]
    #[test]
    fn euler_rotation()
    {
        panic!()
    }

    #[ignore]
    #[test]
    fn look_at()
    {
        panic!()
    }
}

