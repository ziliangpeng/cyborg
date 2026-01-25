use math::digit_factorial_sum;

pub fn solve(verbose: bool) -> String {
    let matches: Vec<u64> = (3..5_000_000u64)
        .filter(|&n| digit_factorial_sum(n) == n)
        .collect();

    if verbose {
        println!("Numbers: {:?}", matches);
    }

    matches.iter().sum::<u64>().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(true), "40730");
    }
}
