use math::fibonacci;

/// PE2: Even Fibonacci Numbers
/// Sum all even-valued Fibonacci numbers not exceeding 4 million.
pub fn solve(_verbose: bool) -> String {
    fibonacci()
        .take_while(|&n| n <= 4_000_000)
        .filter(|n| n % 2 == 0)
        .sum::<u64>()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "4613732");
    }
}
