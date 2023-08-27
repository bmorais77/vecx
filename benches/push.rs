use any_vec::{any_value::AnyValueWrapper, AnyVec};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use vecx::{TypeErasedVec, TypedValue};

const SIZES: [usize; 5] = [1000, 10_000, 100_000, 1_000_000, 10_000_000];

fn vec_push<T: Clone>(value: T, size: usize) {
    let mut vec = Vec::new();
    for _ in 0..size {
        vec.push(value.clone());
    }
}

fn type_erased_vec_push<T: Clone>(value: T, size: usize) {
    let mut vec = TypeErasedVec::new::<T>();
    for _ in 0..size {
        unsafe {
            let value = TypedValue::new(value.clone());
            vec.push(value);
        }
    }
}

fn any_vec_push<T: Clone + 'static>(value: T, size: usize) {
    let mut vec: AnyVec = AnyVec::new::<T>();
    for _ in 0..size {
        unsafe {
            let value = AnyValueWrapper::new(value.clone());
            vec.push_unchecked(value);
        }
    }
}

pub fn bench_push(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("push_32");
        for size in SIZES.into_iter() {
            group.bench_with_input(
                BenchmarkId::new("TypeErasedVec", size),
                &size,
                |b, &size| {
                    b.iter(|| type_erased_vec_push::<u32>(42, size));
                },
            );

            group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
                b.iter(|| vec_push::<u32>(42, size));
            });

            group.bench_with_input(BenchmarkId::new("AnyVec", size), &size, |b, &size| {
                b.iter(|| any_vec_push::<u32>(42, size));
            });
        }
    }

    {
        let mut group = c.benchmark_group("push_256");
        for size in SIZES.into_iter() {
            group.bench_with_input(
                BenchmarkId::new("TypeErasedVec", size),
                &size,
                |b, &size| {
                    b.iter(|| type_erased_vec_push::<(u128, u128)>((42, 42), size));
                },
            );

            group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
                b.iter(|| vec_push::<(u128, u128)>((42, 42), size));
            });

            group.bench_with_input(BenchmarkId::new("AnyVec", size), &size, |b, &size| {
                b.iter(|| any_vec_push::<(u128, u128)>((42, 42), size));
            });
        }
    }

    {
        let mut group = c.benchmark_group("push_512");
        for size in SIZES.into_iter() {
            group.bench_with_input(
                BenchmarkId::new("TypeErasedVec", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        type_erased_vec_push::<(u128, u128, u128, u128)>((42, 42, 42, 42), size)
                    });
                },
            );

            group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
                b.iter(|| vec_push::<(u128, u128, u128, u128)>((42, 42, 42, 42), size));
            });

            group.bench_with_input(BenchmarkId::new("AnyVec", size), &size, |b, &size| {
                b.iter(|| any_vec_push::<(u128, u128, u128, u128)>((42, 42, 42, 42), size));
            });
        }
    }
}

criterion_group!(benches, bench_push);
criterion_main!(benches);
