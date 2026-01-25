//! Prime number utilities including sieves, iterators, and factorization.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Iterator that generates prime numbers.
///
/// Maintains an internal list of found primes and checks each candidate
/// only against primes up to its square root.
///
/// # Example
///
/// ```
/// use math::prime::Primes;
///
/// let first_ten: Vec<u64> = Primes::new().take(10).collect();
/// assert_eq!(first_ten, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
/// ```
pub struct Primes {
    found: Vec<u64>,
    candidate: u64,
}

impl Primes {
    pub fn new() -> Self {
        Primes {
            found: Vec::new(),
            candidate: 2,
        }
    }
}

impl Default for Primes {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for Primes {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let c = self.candidate;
            self.candidate += if c == 2 { 1 } else { 2 };

            let sqrt_c = (c as f64).sqrt() as u64;
            let is_prime = self.found.iter()
                .take_while(|&&p| p <= sqrt_c)
                .all(|&p| c % p != 0);

            if is_prime {
                self.found.push(c);
                return Some(c);
            }
        }
    }
}

/// Returns an iterator over prime numbers: 2, 3, 5, 7, 11, ...
///
/// # Example
///
/// ```
/// use math::prime::primes;
///
/// let first_five: Vec<u64> = primes().take(5).collect();
/// assert_eq!(first_five, vec![2, 3, 5, 7, 11]);
/// ```
pub fn primes() -> Primes {
    Primes::new()
}

/// Returns the nth prime number (1-indexed: nth_prime(1) = 2, nth_prime(2) = 3, ...).
///
/// Returns None if n is 0.
///
/// # Example
///
/// ```
/// use math::prime::nth_prime;
///
/// assert_eq!(nth_prime(1), Some(2));
/// assert_eq!(nth_prime(6), Some(13));
/// assert_eq!(nth_prime(0), None);
/// ```
pub fn nth_prime(n: usize) -> Option<u64> {
    if n == 0 {
        return None;
    }
    Primes::new().nth(n - 1)
}

/// Returns the largest prime factor of n.
///
/// Returns None if n < 2.
///
/// # Example
///
/// ```
/// use math::prime::largest_prime_factor;
///
/// assert_eq!(largest_prime_factor(84), Some(7)); // 84 = 2^2 * 3 * 7
/// assert_eq!(largest_prime_factor(2), Some(2));
/// assert_eq!(largest_prime_factor(1), None);
/// ```
pub fn largest_prime_factor(mut n: u64) -> Option<u64> {
    if n < 2 {
        return None;
    }

    let mut largest = None;
    let mut factor = 2;

    while factor * factor <= n {
        while n.is_multiple_of(factor) {
            largest = Some(factor);
            n /= factor;
        }
        factor += 1;
    }
    if n > 1 {
        largest = Some(n);
    }
    largest
}

/// Count the number of divisors of n using prime factorization.
///
/// Takes a slice of primes that must include all primes up to sqrt(n).
/// Panics if the last prime in the slice is less than sqrt(n).
///
/// # Example
///
/// ```
/// use math::prime::count_divisors;
///
/// let primes = vec![2, 3, 5, 7];
/// assert_eq!(count_divisors(28, &primes), 6); // 1, 2, 4, 7, 14, 28
/// assert_eq!(count_divisors(12, &primes), 6); // 1, 2, 3, 4, 6, 12
/// ```
pub fn count_divisors(mut n: u64, primes: &[u64]) -> u32 {
    if n <= 1 {
        return n as u32;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    let last_prime = *primes.last().expect("primes slice cannot be empty");
    if last_prime < sqrt_n {
        panic!(
            "primes slice insufficient: last prime {} < sqrt({})",
            last_prime, n
        );
    }

    let mut count = 1u32;

    for &p in primes {
        if p * p > n {
            break;
        }
        let mut exp = 0;
        while n % p == 0 {
            exp += 1;
            n /= p;
        }
        if exp > 0 {
            count *= exp + 1;
        }
    }

    // If remainder > 1, it's a prime factor with exponent 1
    if n > 1 {
        count *= 2;
    }

    count
}

/// Check if a number is prime.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (3..=sqrt_n).step_by(2) {
        if n.is_multiple_of(i) {
            return false;
        }
    }
    true
}

/// Finds all prime numbers less than a given limit `n`. (Sequential)
pub fn primes_below(n: u32) -> Vec<u32> {
    if n < 3 {
        return vec![];
    }
    let n_usize = n as usize;
    let mut sieve = vec![true; n_usize];
    sieve[0] = false;
    sieve[1] = false;

    for i in 2..=(n_usize as f64).sqrt() as usize {
        if sieve[i] {
            for multiple in (i * i..n_usize).step_by(i) {
                sieve[multiple] = false;
            }
        }
    }

    (2..n_usize)
        .filter_map(|i| if sieve[i] { Some(i as u32) } else { None })
        .collect()
}

/// Finds all prime numbers less than `n` by parallelizing the outer loop.
pub fn primes_below_parallel_outer(n: u32) -> Vec<u32> {
    if n < 3 {
        return vec![];
    }
    let n_usize = n as usize;
    let sieve: Vec<AtomicBool> = (0..n_usize).map(|_| AtomicBool::new(true)).collect();
    sieve[0].store(false, Ordering::Relaxed);
    sieve[1].store(false, Ordering::Relaxed);

    let sqrt_n = (n_usize as f64).sqrt() as usize;

    (2..=sqrt_n).into_par_iter().for_each(|i| {
        if sieve[i].load(Ordering::Relaxed) {
            for multiple in (i * i..n_usize).step_by(i) {
                sieve[multiple].store(false, Ordering::Relaxed);
            }
        }
    });

    (2..n_usize)
        .filter_map(|i| {
            if sieve[i].load(Ordering::Relaxed) {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect()
}

/// Finds all prime numbers less than `n` by parallelizing the inner loop.
pub fn primes_below_parallel_inner(n: u32) -> Vec<u32> {
    if n < 3 {
        return vec![];
    }
    let n_usize = n as usize;
    let sieve: Vec<AtomicBool> = (0..n_usize).map(|_| AtomicBool::new(true)).collect();
    sieve[0].store(false, Ordering::Relaxed);
    sieve[1].store(false, Ordering::Relaxed);

    let sqrt_n = (n_usize as f64).sqrt() as usize;

    for i in 2..=sqrt_n {
        if sieve[i].load(Ordering::Relaxed) {
            // Corrected limit calculation: ensures `multiple` is strictly less than `n_usize`.
            let limit = (n_usize.saturating_sub(1).saturating_sub(i * i)) / i;
            (0..=limit).into_par_iter().for_each(|k| {
                let multiple = i * i + k * i;
                sieve[multiple].store(false, Ordering::Relaxed);
            });
        }
    }

    (2..n_usize)
        .filter_map(|i| {
            if sieve[i].load(Ordering::Relaxed) {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primes_below_10() {
        let expected = vec![2, 3, 5, 7];
        assert_eq!(primes_below(10), expected);
        assert_eq!(primes_below_parallel_outer(10), expected);
        assert_eq!(primes_below_parallel_inner(10), expected);
    }

    #[test]
    fn test_primes_below_30() {
        let expected = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        assert_eq!(primes_below(30), expected);
        assert_eq!(primes_below_parallel_outer(30), expected);
        assert_eq!(primes_below_parallel_inner(30), expected);
    }

    #[test]
    fn test_primes_below_2() {
        assert_eq!(primes_below(2), vec![]);
        assert_eq!(primes_below_parallel_outer(2), vec![]);
        assert_eq!(primes_below_parallel_inner(2), vec![]);
    }

    #[test]
    fn test_primes_below_3() {
        assert_eq!(primes_below(3), vec![2]);
        assert_eq!(primes_below_parallel_outer(3), vec![2]);
        assert_eq!(primes_below_parallel_inner(3), vec![2]);
    }

    #[test]
    fn test_primes_below_0_and_1() {
        assert_eq!(primes_below(0), vec![]);
        assert_eq!(primes_below(1), vec![]);
        assert_eq!(primes_below_parallel_outer(0), vec![]);
        assert_eq!(primes_below_parallel_outer(1), vec![]);
        assert_eq!(primes_below_parallel_inner(0), vec![]);
        assert_eq!(primes_below_parallel_inner(1), vec![]);
    }

    #[test]
    fn test_consistency_across_implementations() {
        // Test a range of N values to ensure consistency
        let ns = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

        for &n in ns {
            let sequential_result = primes_below(n);
            let parallel_outer_result = primes_below_parallel_outer(n);
            let parallel_inner_result = primes_below_parallel_inner(n);

            assert_eq!(
                sequential_result, parallel_outer_result,
                "Mismatch for n={} between sequential and parallel_outer",
                n
            );
            assert_eq!(
                sequential_result, parallel_inner_result,
                "Mismatch for n={} between sequential and parallel_inner",
                n
            );
        }
    }

    #[test]
    fn test_primes_iterator() {
        let first_ten: Vec<u64> = Primes::new().take(10).collect();
        assert_eq!(first_ten, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_primes_iterator_vs_sieve() {
        // Compare iterator output against sieve for primes below 100,000
        let limit = 100_000u32;
        let sieve_primes = primes_below(limit);
        let iter_primes: Vec<u64> = Primes::new()
            .take_while(|&p| p < limit as u64)
            .collect();
        let sieve_as_u64: Vec<u64> = sieve_primes.iter().map(|&p| p as u64).collect();
        assert_eq!(iter_primes, sieve_as_u64);
    }

    #[test]
    fn test_primes_function() {
        let first_five: Vec<u64> = primes().take(5).collect();
        assert_eq!(first_five, vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(nth_prime(0), None);
        assert_eq!(nth_prime(1), Some(2));
        assert_eq!(nth_prime(2), Some(3));
        assert_eq!(nth_prime(6), Some(13));
        assert_eq!(nth_prime(1000), Some(7919));
    }

    #[test]
    fn test_largest_prime_factor() {
        assert_eq!(largest_prime_factor(0), None);
        assert_eq!(largest_prime_factor(1), None);
        assert_eq!(largest_prime_factor(2), Some(2));
        assert_eq!(largest_prime_factor(84), Some(7)); // 84 = 2^2 * 3 * 7
        assert_eq!(largest_prime_factor(100), Some(5));
    }

    #[test]
    fn test_count_divisors() {
        let primes: Vec<u64> = Primes::new().take(100).collect();
        assert_eq!(count_divisors(1, &primes), 1);
        assert_eq!(count_divisors(2, &primes), 2); // 1, 2
        assert_eq!(count_divisors(6, &primes), 4); // 1, 2, 3, 6
        assert_eq!(count_divisors(12, &primes), 6); // 1, 2, 3, 4, 6, 12
        assert_eq!(count_divisors(28, &primes), 6); // 1, 2, 4, 7, 14, 28
        assert_eq!(count_divisors(36, &primes), 9); // 1, 2, 3, 4, 6, 9, 12, 18, 36
    }

    #[test]
    #[should_panic(expected = "primes slice insufficient")]
    fn test_count_divisors_insufficient_primes() {
        let primes = vec![2, 3]; // sqrt(100) = 10, but max prime is 3
        count_divisors(100, &primes);
    }
}
