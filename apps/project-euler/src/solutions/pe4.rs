/// PE4: Largest Palindrome Product
/// Find the largest palindrome made from the product of two 3-digit numbers.
pub fn solve(_verbose: bool) -> String {
    let mut largest = 0;

    for i in (100..1000).rev() {
        for j in (100..=i).rev() {
            let product = i * j;
            if product <= largest {
                break;
            }
            if is_palindrome(product) {
                largest = product;
            }
        }
    }
    largest.to_string()
}

fn is_palindrome(n: u32) -> bool {
    let s = n.to_string();
    s == s.chars().rev().collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "906609");
    }
}
