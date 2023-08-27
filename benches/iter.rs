use any_vec::{any_value::AnyValueWrapper, AnyVec};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vecx::{TypeErasedVec, TypedValue};

const SIZES: [usize; 5] = [1000, 10_000, 100_000, 1_000_000, 10_000_000];

fn workload<I: Iterator>(mut iter: I) {
    let mut curr = iter.next();
    let mut count = 0;
    while curr.is_some() {
        curr = black_box(iter.next());
        count += black_box(1);
    }

    black_box(count);
}

fn build_vec(n: usize) -> Vec<usize> {
    let mut vec = Vec::with_capacity(n);
    for i in 0..n {
        vec.push(i);
    }
    vec
}

fn build_type_erased_vec(n: usize) -> TypeErasedVec {
    let mut vec = TypeErasedVec::new::<usize>();
    vec.reserve(n);

    for i in 0..n {
        unsafe {
            vec.push(TypedValue::new(i));
        }
    }

    vec
}

fn build_any_vec(n: usize) -> AnyVec {
    let mut vec = AnyVec::with_capacity::<usize>(n);
    for i in 0..n {
        vec.push(AnyValueWrapper::new(i));
    }

    vec
}

pub fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");
    for size in SIZES.into_iter() {
        group.bench_with_input(
            BenchmarkId::new("TypeErasedVec", size),
            &size,
            |b, &size| {
                let vec = build_type_erased_vec(size);
                b.iter(|| workload(vec.iter()));
            },
        );

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            let vec = build_vec(size);
            b.iter(|| workload(vec.iter()));
        });

        group.bench_with_input(BenchmarkId::new("AnyVec", size), &size, |b, &size| {
            let vec = build_any_vec(size);
            b.iter(|| workload(vec.iter()));
        });
    }
}

criterion_group!(benches, bench_iter);
criterion_main!(benches);
