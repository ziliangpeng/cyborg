//! Sieve of Eratosthenes prime number generator.

/// Finds all prime numbers less than a given limit `n`.
///
/// Implements the Sieve of Eratosthenes algorithm.
///
/// # Panics
///
/// This function will panic if `n` is larger than can be represented by `usize`,
/// as it needs to allocate a boolean vector of this size.
///
/// # Examples
///
/// ```
/// // Note: This is a doc test that will be run by `cargo test`.
/// // To use this in another crate, you would `use math::primes_below;`
/// // assert_eq!(math::primes_below(10), vec![2, 3, 5, 7]);
/// ```
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

    sieve
        .into_iter()
        .enumerate()
        .filter_map(|(num, is_prime)| if is_prime { Some(num as u32) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primes_below_10() {
        assert_eq!(primes_below(10), vec![2, 3, 5, 7]);
    }

    #[test]
    fn test_primes_below_30() {
        assert_eq!(primes_below(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_primes_below_2() {
        assert_eq!(primes_below(2), vec![]);
    }
    
    #[test]
    fn test_primes_below_3() {
        assert_eq!(primes_below(3), vec![2]);
    }

    #[test]
    fn test_primes_below_0_and_1() {
        assert_eq!(primes_below(0), vec![]);
        assert_eq!(primes_below(1), vec![]);
    }
}
