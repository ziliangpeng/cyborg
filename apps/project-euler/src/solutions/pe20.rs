use math::big_factorial;

pub fn solve(verbose: bool) -> String {
    let result = big_factorial(100);

    if verbose {
        println!("100! = {}", result);
    }

    let digit_sum: u32 = result
        .to_string()
        .chars()
        .map(|c| c.to_digit(10).unwrap())
        .sum();

    digit_sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_factorial_digit_sum() {
        // 10! = 3628800, digit sum = 3+6+2+8+8+0+0 = 27
        let result = math::big_factorial(10);
        assert_eq!(result.to_string(), "3628800");
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "648");
    }
}
