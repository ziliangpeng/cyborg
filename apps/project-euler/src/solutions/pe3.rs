/// PE3: Largest Prime Factor
/// Find the largest prime factor of 600851475143.
pub fn solve(_verbose: bool) -> String {
    let mut n: u64 = 600851475143;
    let mut largest = 1;
    let mut factor = 2;

    while factor * factor <= n {
        while n % factor == 0 {
            largest = factor;
            n /= factor;
        }
        factor += 1;
    }
    if n > 1 {
        largest = n;
    }
    largest.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "6857");
    }
}
