/// PE7: 10001st Prime
/// Find the 10001st prime number.
pub fn solve(_verbose: bool) -> String {
    let mut count = 0;
    let mut n = 2;
    while count < 10001 {
        if is_prime(n) {
            count += 1;
        }
        if count < 10001 {
            n += 1;
        }
    }
    n.to_string()
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "104743");
    }
}
