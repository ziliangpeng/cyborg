use crate::BigInt;

/// Compute factorial of n (n!)
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}

/// Compute factorial of n (n!) as BigInt for arbitrary precision
pub fn big_factorial(n: u32) -> BigInt {
    let mut result = BigInt::from_u32(1);
    for i in 1..=n {
        result = result * i;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }
}
