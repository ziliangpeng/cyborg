use math::nth_prime;

/// PE7: 10001st Prime
/// Find the 10001st prime number.
pub fn solve(_verbose: bool) -> String {
    nth_prime(10001).unwrap().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "104743");
    }
}
