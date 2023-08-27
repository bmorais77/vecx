use any_vec::AnyVec;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vecx::TypeErasedVec;

const SIZES: [usize; 5] = [1000, 10_000, 100_000, 1_000_000, 10_000_000];

fn vec_reserve(size: usize) {
    let mut vec = Vec::<i32>::new();
    vec.reserve(size);
    black_box(vec);
}

fn type_erased_vec_reserve(size: usize) {
    let mut vec = TypeErasedVec::new::<i32>();
    vec.reserve(size);
    black_box(vec);
}

fn any_vec_reserve(size: usize) {
    let mut vec: AnyVec = AnyVec::new::<i32>();
    vec.reserve(size);
    black_box(vec);
}

pub fn bench_reserve(c: &mut Criterion) {
    let mut group = c.benchmark_group("reserve");
    for size in SIZES.into_iter() {
        group.bench_with_input(
            BenchmarkId::new("TypeErasedVec", size),
            &size,
            |b, &size| {
                b.iter(|| vec_reserve(size));
            },
        );

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter(|| type_erased_vec_reserve(size));
        });

        group.bench_with_input(BenchmarkId::new("AnyVec", size), &size, |b, &size| {
            b.iter(|| any_vec_reserve(size));
        });
    }
}

criterion_group!(benches, bench_reserve);
criterion_main!(benches);
