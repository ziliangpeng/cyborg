use math::primes_below;

pub fn solve(verbose: bool) -> String {
    let _ = verbose;
    let primes = primes_below(2_000_000);
    let sum: u64 = primes.iter().map(|&p| p as u64).sum();
    sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "142913828922");
    }
}
