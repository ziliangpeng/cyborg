use crate::factorial::factorial;

/// Compute the sum of factorials of each digit in n.
///
/// Example: digit_factorial_sum(145) = 1! + 4! + 5! = 1 + 24 + 120 = 145
pub fn digit_factorial_sum(mut n: u64) -> u64 {
    let mut sum = 0;
    while n > 0 {
        let digit = n % 10;
        sum += factorial(digit);
        n /= 10;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digit_factorial_sum() {
        assert_eq!(digit_factorial_sum(145), 145); // 1! + 4! + 5! = 1 + 24 + 120 = 145
        assert_eq!(digit_factorial_sum(169), 363601); // 1! + 6! + 9! = 1 + 720 + 362880
        assert_eq!(digit_factorial_sum(0), 0);
    }
}
