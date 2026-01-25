/// PE5: Smallest Multiple
/// Find the smallest positive number evenly divisible by all numbers from 1 to 20.
/// This is the LCM of 1..20.
pub fn solve(_verbose: bool) -> String {
    let result = (1u64..=20).fold(1, lcm);
    result.to_string()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "232792560");
    }
}
