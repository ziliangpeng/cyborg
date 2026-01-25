use math::{Primes, TriangleNumbers, count_divisors};

pub fn solve(verbose: bool) -> String {
    let mut primes: Vec<u64> = Vec::new();
    let mut prime_iter = Primes::new();

    for t in TriangleNumbers::new() {
        // Ensure we have enough primes (up to sqrt(t))
        let sqrt_t = (t as f64).sqrt() as u64;
        while primes.last().map_or(true, |&p| p < sqrt_t) {
            primes.push(prime_iter.next().unwrap());
        }

        let divisor_count = count_divisors(t, &primes);

        if verbose && divisor_count > 400 {
            println!("T({}) = {}, divisors = {}", primes.len(), t, divisor_count);
        }

        if divisor_count > 500 {
            return t.to_string();
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_numbers() {
        use math::TriangleNumbers;
        let nums: Vec<u64> = TriangleNumbers::new().take(10).collect();
        println!("First 10 triangle numbers: {:?}", nums);
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "76576500");
    }
}
