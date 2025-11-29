use criterion::{criterion_group, criterion_main, Criterion};
use math::{
    primes_below, primes_below_parallel_inner, primes_below_parallel_outer,
};

fn run_all_benchmarks(c: &mut Criterion) {
    let mut group_10k = c.benchmark_group("primes_below_n_10000");
    group_10k.bench_function("sequential", |b| b.iter(|| primes_below(10_000)));
    group_10k.bench_function("parallel_outer", |b| {
        b.iter(|| primes_below_parallel_outer(10_000))
    });
    group_10k.bench_function("parallel_inner", |b| {
        b.iter(|| primes_below_parallel_inner(10_000))
    });
    group_10k.finish();

    let mut group_1m = c.benchmark_group("primes_below_n_1000000");
    group_1m.bench_function("sequential", |b| b.iter(|| primes_below(1_000_000)));
    group_1m.bench_function("parallel_outer", |b| {
        b.iter(|| primes_below_parallel_outer(1_000_000))
    });
    group_1m.bench_function("parallel_inner", |b| {
        b.iter(|| primes_below_parallel_inner(1_000_000))
    });
    group_1m.finish();

    let mut group_10m = c.benchmark_group("primes_below_n_10000000");
    group_10m.sample_size(10);
    group_10m.bench_function("sequential", |b| b.iter(|| primes_below(10_000_000)));
    group_10m.bench_function("parallel_outer", |b| {
        b.iter(|| primes_below_parallel_outer(10_000_000))
    });
    group_10m.bench_function("parallel_inner", |b| {
        b.iter(|| primes_below_parallel_inner(10_000_000))
    });
    group_10m.finish();

    let mut group_100m = c.benchmark_group("primes_below_n_100000000");
    group_100m.sample_size(10); // Changed from 5 to 10
    group_100m.bench_function("sequential", |b| b.iter(|| primes_below(100_000_000)));
    group_100m.bench_function("parallel_outer", |b| {
        b.iter(|| primes_below_parallel_outer(100_000_000))
    });
    group_100m.bench_function("parallel_inner", |b| {
        b.iter(|| primes_below_parallel_inner(100_000_000))
    });
    group_100m.finish();
}

criterion_group!(benches, run_all_benchmarks);
criterion_main!(benches);
