use math::largest_prime_factor;

/// PE3: Largest Prime Factor
/// Find the largest prime factor of 600851475143.
pub fn solve(_verbose: bool) -> String {
    largest_prime_factor(600851475143).unwrap().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "6857");
    }
}
