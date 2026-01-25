use math::{digit_factorial_sum, find_chain_length};

pub fn solve(_verbose: bool) -> String {
    let count = (1..1_000_000u64)
        .filter(|&n| find_chain_length(n, digit_factorial_sum) == 60)
        .count();
    count.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_length() {
        // 69 → 363600 → 1454 → 169 → 363601 (→ 1454)
        assert_eq!(find_chain_length(69u64, digit_factorial_sum), 5);
        // 145 → 145 (loop immediately)
        assert_eq!(find_chain_length(145u64, digit_factorial_sum), 1);
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "402");
    }
}
