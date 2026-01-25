use std::fmt;
use std::ops::{Add, Mul};
use std::str::FromStr;

/// A big integer represented as a vector of digits (least significant first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigInt {
    digits: Vec<u8>,
}

impl BigInt {
    /// Create a BigInt representing zero.
    pub fn zero() -> Self {
        BigInt { digits: vec![0] }
    }

    /// Create a BigInt from a u32.
    pub fn from_u32(mut n: u32) -> Self {
        if n == 0 {
            return BigInt::zero();
        }
        let mut digits = Vec::new();
        while n > 0 {
            digits.push((n % 10) as u8);
            n /= 10;
        }
        BigInt { digits }
    }
}

impl FromStr for BigInt {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let digits: Vec<u8> = s
            .chars()
            .rev()
            .filter_map(|c| c.to_digit(10).map(|d| d as u8))
            .collect();
        Ok(BigInt { digits })
    }
}

impl Add for BigInt {
    type Output = BigInt;

    fn add(self, other: BigInt) -> BigInt {
        &self + &other
    }
}

impl Add for &BigInt {
    type Output = BigInt;

    fn add(self, other: &BigInt) -> BigInt {
        let mut result = Vec::new();
        let mut carry = 0u8;
        let max_len = self.digits.len().max(other.digits.len());

        for i in 0..max_len {
            let a = self.digits.get(i).copied().unwrap_or(0);
            let b = other.digits.get(i).copied().unwrap_or(0);
            let sum = a + b + carry;
            result.push(sum % 10);
            carry = sum / 10;
        }

        if carry > 0 {
            result.push(carry);
        }

        BigInt { digits: result }
    }
}

impl Mul<u32> for BigInt {
    type Output = BigInt;

    fn mul(self, other: u32) -> BigInt {
        &self * other
    }
}

impl Mul<u32> for &BigInt {
    type Output = BigInt;

    fn mul(self, other: u32) -> BigInt {
        let mut result = Vec::new();
        let mut carry = 0u32;

        for &d in &self.digits {
            let product = (d as u32) * other + carry;
            result.push((product % 10) as u8);
            carry = product / 10;
        }

        while carry > 0 {
            result.push((carry % 10) as u8);
            carry /= 10;
        }

        BigInt { digits: result }
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.digits.is_empty() {
            return write!(f, "0");
        }
        for &d in self.digits.iter().rev() {
            write!(f, "{}", d)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() {
        let n: BigInt = "12345".parse().unwrap();
        assert_eq!(n.digits, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_display() {
        let n: BigInt = "12345".parse().unwrap();
        assert_eq!(format!("{}", n), "12345");
    }

    #[test]
    fn test_add_simple() {
        let a: BigInt = "123".parse().unwrap();
        let b: BigInt = "456".parse().unwrap();
        let sum = a + b;
        assert_eq!(format!("{}", sum), "579");
    }

    #[test]
    fn test_add_with_carry() {
        let a: BigInt = "999".parse().unwrap();
        let b: BigInt = "1".parse().unwrap();
        let sum = a + b;
        assert_eq!(format!("{}", sum), "1000");
    }

    #[test]
    fn test_add_different_lengths() {
        let a: BigInt = "12345".parse().unwrap();
        let b: BigInt = "99".parse().unwrap();
        let sum = a + b;
        assert_eq!(format!("{}", sum), "12444");
    }

    #[test]
    fn test_from_u32() {
        let n = BigInt::from_u32(12345);
        assert_eq!(format!("{}", n), "12345");
    }

    #[test]
    fn test_mul_u32() {
        let a: BigInt = "123".parse().unwrap();
        let product = a * 4;
        assert_eq!(format!("{}", product), "492");
    }

    #[test]
    fn test_mul_u32_with_carry() {
        let a: BigInt = "999".parse().unwrap();
        let product = a * 9;
        assert_eq!(format!("{}", product), "8991");
    }
}
