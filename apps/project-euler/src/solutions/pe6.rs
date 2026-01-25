/// PE6: Sum Square Difference
/// Find the difference between the square of the sum and the sum of squares
/// for the first 100 natural numbers.
pub fn solve(_verbose: bool) -> String {
    let n: u64 = 100;
    let sum = n * (n + 1) / 2;
    let sum_of_squares = n * (n + 1) * (2 * n + 1) / 6;
    let square_of_sum = sum * sum;
    (square_of_sum - sum_of_squares).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "25164150");
    }
}
