fn digit_fifth_power_sum(mut n: u64) -> u64 {
    let mut sum = 0;
    while n > 0 {
        let d = n % 10;
        sum += d.pow(5);
        n /= 10;
    }
    sum
}

pub fn solve(verbose: bool) -> String {
    let matches: Vec<u64> = (2..1_000_000u64)
        .filter(|&n| digit_fifth_power_sum(n) == n)
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
    fn test_digit_fifth_power_sum() {
        // 4150 = 4^5 + 1^5 + 5^5 + 0^5 = 1024 + 1 + 3125 + 0 = 4150
        assert_eq!(digit_fifth_power_sum(4150), 4150);
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "443839");
    }
}
