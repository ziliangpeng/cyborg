use math::proper_divisors;

pub fn solve(verbose: bool) -> String {
    let mut sum = 0u64;

    for a in 1..10000 {
        let b: u64 = proper_divisors(a).iter().sum();
        let c: u64 = proper_divisors(b).iter().sum();

        if c == a && a != b {
            if verbose {
                println!("{} and {} are amicable", a, b);
            }
            sum += a;
        }
    }

    sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "31626");
    }
}
