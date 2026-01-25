/// PE9: Special Pythagorean Triplet
/// Find the product abc where a + b + c = 1000 and a² + b² = c².
pub fn solve(_verbose: bool) -> String {
    for a in 1..500 {
        for b in (a + 1)..(1000 - a) / 2 {
            let c = 1000 - a - b;
            if a * a + b * b == c * c {
                return (a * b * c).to_string();
            }
        }
    }
    "Not found".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "31875000");
    }
}
