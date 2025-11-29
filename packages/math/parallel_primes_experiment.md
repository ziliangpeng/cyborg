# Parallel Prime Generation Experiment

## Introduction

This document details an experiment conducted to compare different strategies for parallelizing the Sieve of Eratosthenes algorithm for finding prime numbers. Three implementations were developed and benchmarked to understand their performance characteristics.

## Implementations Tested

1.  **`primes_below` (Sequential):** The baseline, single-threaded implementation of the Sieve of Eratosthenes.
2.  **`primes_below_parallel_outer`:** An attempt to parallelize the sieve by making the *outer loop* (iterating through prime factors to mark multiples) run in parallel using `rayon` and `AtomicBool` for thread-safe access to the sieve.
3.  **`primes_below_parallel_inner`:** An attempt to parallelize the sieve by making the *inner loop* (marking multiples for a single prime factor) run in parallel, also using `rayon` and `AtomicBool`.

## Implementation Challenges for `primes_below_parallel_inner`

Initially, `primes_below_parallel_inner` faced compilation errors due to `rayon`'s inability to parallelize `step_by` iterators directly. This was resolved by re-implementing the inner loop's parallelization to use a finite range over which `rayon` could operate, calculating the specific multiples inside the parallel task. Further, a runtime panic due to an off-by-one error in calculating array bounds was identified and fixed, ensuring correct indexing.

## Benchmark Methodology

The three implementations were benchmarked using the `criterion` library for various input sizes (`n`). Benchmarks were run on a multi-core system to observe the effects of parallelism.

## Benchmark Results Summary

Here's a summary of the median runtimes from the benchmark, comparing the performance across different `n` values.

| n             | `sequential` (Time) | `parallel_outer` (Time) | `parallel_inner` (Time) | `parallel_outer` Speedup | `parallel_inner` Speedup |
| :------------ | :------------------ | :---------------------- | :---------------------- | :----------------------- | :----------------------- |
| **10,000**    | 9.07 µs             | 47.41 µs                | 1.06 ms                 | 0.19x (slower)           | 0.008x (much slower)     |
| **1,000,000** | 1.50 ms             | 1.47 ms                 | 13.02 ms                | 1.02x (slight gain)      | 0.11x (slower)           |
| **10,000,000**| 16.25 ms            | 13.57 ms                | 80.92 ms                | 1.20x (20% faster)       | 0.20x (slower)           |
| **100,000,000**| 352.33 ms           | **173.72 ms**           | 338.48 ms               | **2.03x (2x faster)**    | 1.04x (slight gain)      |

## Analysis and Explanation of Results

*   **`primes_below_parallel_inner` (Parallelizing the Inner Loop):** This implementation consistently performed the worst, being significantly slower than even the sequential version for most input sizes. This confirms that parallelizing the inner loop is inefficient for the Sieve of Eratosthenes. The overhead of creating and managing a multitude of fine-grained parallel tasks (creating a new parallel iterator for each `i` in the outer loop) far outweighs the small amount of work each task performs.

*   **`primes_below_parallel_outer` (Parallelizing the Outer Loop):** This implementation showed the best performance for larger input sizes.
    *   For small `n` (e.g., 10,000), it was slower than sequential due to the fixed overhead of `rayon`.
    *   As `n` increased, the performance gains became significant, with a remarkable **2x speedup** for `n=100,000,000`.
    *   This strategy effectively distributes the "coarse-grained" work of marking multiples for different prime factors across multiple CPU cores, allowing the parallelization benefits to overcome the overhead.

*   **`primes_below` (Sequential):** Served as a strong baseline, performing best for very small `n` where parallelism overhead is not justified.

## Conclusion

The experiment clearly demonstrates that for the Sieve of Eratosthenes algorithm in Rust, **parallelizing the outer loop (`primes_below_parallel_outer`) is the optimal strategy for performance gains on multi-core processors, especially as the input size increases.** This method efficiently utilizes available CPU resources. Parallelizing the inner loop, however, introduces too much overhead and should be avoided.
