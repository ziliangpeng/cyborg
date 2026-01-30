fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

pub fn solve(verbose: bool) -> String {
    let mut available: Vec<u64> = (0..10).collect();
    let mut remaining = 999_999u64;
    let mut result = String::new();

    for i in 0..10 {
        let num_remaining_digits = 9 - i;
        let fact = factorial(num_remaining_digits);
        let index = remaining / fact;
        remaining %= fact;

        let digit = available.remove(index as usize);
        result.push_str(&digit.to_string());

        if verbose {
            println!(
                "Step {}: fact={}, index={}, picked={}, remaining={}",
                i + 1,
                fact,
                index,
                digit,
                remaining
            );
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "2783915460");
    }
}
