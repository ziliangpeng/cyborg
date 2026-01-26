use math::BigFibonacci;

pub fn solve(verbose: bool) -> String {
    for (i, fib) in BigFibonacci::new().enumerate() {
        let digit_count = fib.to_string().len();
        if verbose && digit_count >= 990 {
            println!("F({}) has {} digits", i + 1, digit_count);
        }
        if digit_count >= 1000 {
            return (i + 1).to_string();
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_12_digit_fibonacci() {
        // F(12) = 144 is the first with 3 digits
        for (i, fib) in math::BigFibonacci::new().enumerate() {
            if fib.to_string().len() >= 3 {
                assert_eq!(i + 1, 12);
                break;
            }
        }
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "4782");
    }
}
