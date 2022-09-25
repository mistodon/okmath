use criterion::{black_box, criterion_group, criterion_main, Criterion};
use okmath::*;

fn bench_map(c: &mut Criterion) {
    let a = vec4(1.1, 20.2, 300.3, 4000.4);
    c.bench_function(&format!("Vec4::map"), |m| {
        m.iter(|| black_box(a.clone()).map(f32::round))
    });
}

fn bench_zipmap(c: &mut Criterion) {
    let a = vec4(1.1, 20.2, 300.3, 4000.4);
    let b = vec4(2.1, 30.2, 400.3, 5000.4);
    c.bench_function(&format!("Vec4::zipmap"), |m| {
        m.iter(|| black_box(a.clone()).zipmap(black_box(b.clone()), |x, y| x + y))
    });
}

fn bench_transpose(c: &mut Criterion) {
    let t = Mat4::new([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 1., 2., 3.],
        [4., 5., 6., 7.],
    ]);
    c.bench_function(&format!("Mat4::transpose"), |m| {
        m.iter(|| black_box(t.clone()).transpose())
    });
}

fn bench_identity(c: &mut Criterion) {
    c.bench_function(&format!("Mat4::identity"), |m| {
        m.iter(|| Mat4::<f32>::identity())
    });
}

fn bench_translation(c: &mut Criterion) {
    let t = [4., 3., 2., 1.];
    c.bench_function(&format!("Mat4::translation_homogenous"), |m| {
        m.iter(|| Mat4::translation_homogenous(black_box(t.clone())))
    });
}

fn bench_matmul(c: &mut Criterion) {
    let t0 = Mat4::new([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 1., 2., 3.],
        [4., 5., 6., 7.],
    ]);
    let t1 = Mat4::new([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 1., 2., 3.],
        [4., 5., 6., 7.],
    ]);
    c.bench_function(&format!("Mat4::mul::<Mat4>"), |m| {
        m.iter(|| black_box(t0.clone()) * black_box(t1.clone()))
    });
}

fn bench_matmulvec(c: &mut Criterion) {
    let t0 = Mat4::new([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 1., 2., 3.],
        [4., 5., 6., 7.],
    ]);
    let v = vec4(1., 2., 3., 4.);
    c.bench_function(&format!("Mat4::mul::<Vec4>"), |m| {
        m.iter(|| black_box(t0.clone()) * black_box(v.clone()))
    });
}

fn bench_extend_mat(c: &mut Criterion) {
    let t = Mat3::new([[1., 2., 3.], [5., 6., 7.], [9., 1., 2.]]);
    c.bench_function(&format!("Mat3::extend"), |m| {
        m.iter(|| black_box(t).extend())
    });
}

fn bench_retract_mat(c: &mut Criterion) {
    let t = Mat3::new([[1., 2., 3.], [5., 6., 7.], [9., 1., 2.]]);
    c.bench_function(&format!("Mat3::retract"), |m| {
        m.iter(|| black_box(t).retract())
    });
}

criterion_group!(vectors, bench_map, bench_zipmap);
criterion_group!(
    matrices,
    bench_matmul,
    bench_matmulvec,
    bench_transpose,
    bench_identity,
    bench_translation,
    bench_extend_mat,
    bench_retract_mat,
);
criterion_main!(vectors, matrices);
