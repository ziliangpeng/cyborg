use criterion::{criterion_group, criterion_main, Criterion};
use math::primes_below; // Import our function from the 'math' crate

pub fn benchmark_primes_below(c: &mut Criterion) {
    let mut group = c.benchmark_group("primes_below");

    // Benchmark different input sizes
    group.bench_function("n=100", |b| {
        b.iter(|| primes_below(100));
    });

    group.bench_function("n=1_000", |b| {
        b.iter(|| primes_below(1_000));
    });

    group.bench_function("n=10_000", |b| {
        b.iter(|| primes_below(10_000));
    });

    group.bench_function("n=100_000", |b| {
        b.iter(|| primes_below(100_000));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = benchmark_primes_below
}
criterion_main!(benches);
