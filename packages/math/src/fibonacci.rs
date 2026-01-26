use crate::BigInt;

/// Iterator that generates Fibonacci numbers.
///
/// # Example
///
/// ```
/// use math::fibonacci::Fibonacci;
///
/// // Get first 10 Fibonacci numbers
/// let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
/// assert_eq!(fibs, vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
/// ```
pub struct Fibonacci {
    curr: u64,
    next: u64,
}

impl Fibonacci {
    pub fn new() -> Self {
        Fibonacci { curr: 1, next: 1 }
    }

    /// Start with custom initial values (a, b) where the sequence is a, b, a+b, ...
    pub fn with_start(a: u64, b: u64) -> Self {
        Fibonacci { curr: a, next: b }
    }
}

impl Default for Fibonacci {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr;
        self.curr = self.next;
        self.next = curr.checked_add(self.curr)?;
        Some(curr)
    }
}

/// Returns an iterator over Fibonacci numbers starting with 1, 1, 2, 3, 5, ...
///
/// # Example
///
/// ```
/// use math::fibonacci::fibonacci;
///
/// let first_five: Vec<u64> = fibonacci().take(5).collect();
/// assert_eq!(first_five, vec![1, 1, 2, 3, 5]);
/// ```
pub fn fibonacci() -> Fibonacci {
    Fibonacci::new()
}

/// Returns the nth Fibonacci number (1-indexed: fib(1) = 1, fib(2) = 1, fib(3) = 2, ...).
///
/// Returns None if n is 0 or if the result overflows u64.
///
/// # Example
///
/// ```
/// use math::fibonacci::nth_fibonacci;
///
/// assert_eq!(nth_fibonacci(1), Some(1));
/// assert_eq!(nth_fibonacci(10), Some(55));
/// assert_eq!(nth_fibonacci(0), None);
/// ```
pub fn nth_fibonacci(n: usize) -> Option<u64> {
    if n == 0 {
        return None;
    }
    Fibonacci::new().nth(n - 1)
}

/// Iterator that generates Fibonacci numbers as BigInt (arbitrary precision).
///
/// # Example
///
/// ```
/// use math::fibonacci::BigFibonacci;
///
/// let fibs: Vec<String> = BigFibonacci::new().take(10).map(|b| b.to_string()).collect();
/// assert_eq!(fibs, vec!["1", "1", "2", "3", "5", "8", "13", "21", "34", "55"]);
/// ```
pub struct BigFibonacci {
    curr: BigInt,
    next: BigInt,
}

impl BigFibonacci {
    pub fn new() -> Self {
        BigFibonacci {
            curr: BigInt::from_u32(1),
            next: BigInt::from_u32(1),
        }
    }
}

impl Default for BigFibonacci {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for BigFibonacci {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr.clone();
        let new_next = &self.curr + &self.next;
        self.curr = self.next.clone();
        self.next = new_next;
        Some(curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_iterator() {
        let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
        assert_eq!(fibs, vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
    }

    #[test]
    fn test_fibonacci_function() {
        let fibs: Vec<u64> = fibonacci().take(5).collect();
        assert_eq!(fibs, vec![1, 1, 2, 3, 5]);
    }

    #[test]
    fn test_nth_fibonacci() {
        assert_eq!(nth_fibonacci(0), None);
        assert_eq!(nth_fibonacci(1), Some(1));
        assert_eq!(nth_fibonacci(2), Some(1));
        assert_eq!(nth_fibonacci(3), Some(2));
        assert_eq!(nth_fibonacci(10), Some(55));
        assert_eq!(nth_fibonacci(20), Some(6765));
    }

    #[test]
    fn test_fibonacci_with_custom_start() {
        // Lucas numbers: 2, 1, 3, 4, 7, 11, ...
        let lucas: Vec<u64> = Fibonacci::with_start(2, 1).take(6).collect();
        assert_eq!(lucas, vec![2, 1, 3, 4, 7, 11]);
    }
}
