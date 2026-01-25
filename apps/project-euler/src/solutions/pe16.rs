use math::BigInt;

pub fn solve(verbose: bool) -> String {
    let _ = verbose;
    let mut n = BigInt::from_u32(2);

    for _ in 0..999 {
        n = n * 2;
    }

    let s = format!("{}", n);
    let digit_sum: u32 = s.chars().map(|c| c.to_digit(10).unwrap()).sum();

    digit_sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "1366");
    }
}
