use std::time::{Duration, Instant};

use any_vec::{any_value::AnyValueWrapper, AnyVec};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use vecx::{TypeErasedVec, TypedValue};

const SIZES: [usize; 4] = [10, 100, 1000, 10_000];

fn vec_remove<F>(index: F, size: usize) -> Duration
where
    F: Fn(usize) -> usize,
{
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }

    let start = Instant::now();
    {
        for i in (0..size).rev() {
            vec.remove(index(i));
        }
    }
    start.elapsed()
}

fn type_erased_vec_purge<F>(index: F, size: usize) -> Duration
where
    F: Fn(usize) -> usize,
{
    let mut vec = TypeErasedVec::new::<usize>();
    vec.reserve(size);

    for i in 0..size {
        unsafe {
            vec.push(TypedValue::new(i));
        }
    }

    let start = Instant::now();
    {
        for i in (0..size).rev() {
            vec.purge(index(i));
        }
    }
    start.elapsed()
}

fn any_vec_remove<F>(index: F, size: usize) -> Duration
where
    F: Fn(usize) -> usize,
{
    let mut vec: AnyVec = AnyVec::with_capacity::<usize>(size);
    for i in 0..size {
        vec.push(AnyValueWrapper::new(i));
    }

    let start = Instant::now();
    {
        for i in (0..size).rev() {
            vec.remove(index(i));
        }
    }
    start.elapsed()
}

pub fn bench_purge(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("purge_front");
        for size in SIZES.into_iter() {
            bench_custom(
                &mut group,
                BenchmarkId::new("TypeErasedVec", size),
                &size,
                |&size| type_erased_vec_purge(|_| 0, size),
            );

            bench_custom(&mut group, BenchmarkId::new("Vec", size), &size, |&size| {
                vec_remove(|_| 0, size)
            });

            bench_custom(
                &mut group,
                BenchmarkId::new("AnyVec", size),
                &size,
                |&size| any_vec_remove(|_| 0, size),
            );
        }
    }

    {
        let mut group = c.benchmark_group("purge_back");
        for size in SIZES.into_iter() {
            bench_custom(
                &mut group,
                BenchmarkId::new("TypeErasedVec", size),
                &size,
                |&size| type_erased_vec_purge(|i| i, size),
            );

            bench_custom(&mut group, BenchmarkId::new("Vec", size), &size, |&size| {
                vec_remove(|i| i, size)
            });

            bench_custom(
                &mut group,
                BenchmarkId::new("AnyVec", size),
                &size,
                |&size| any_vec_remove(|i| i, size),
            );
        }
    }
}

fn bench_custom<'a, I, F>(
    group: &mut BenchmarkGroup<'a, WallTime>,
    id: BenchmarkId,
    input: &I,
    mut routine: F,
) where
    F: FnMut(&I) -> Duration,
{
    group.bench_with_input(id, input, move |b, input| {
        b.iter_custom(|iters| (0..iters).map(|_| routine(input)).sum())
    });
}

criterion_group!(benches, bench_purge);
criterion_main!(benches);
