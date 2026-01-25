pub fn solve(verbose: bool) -> String {
    let _ = verbose;
    let mut sum = 0;
    for i in 1..1000 {
        if i % 3 == 0 || i % 5 == 0 {
            sum += i;
        }
    }
    sum.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        let answer = solve(false);
        println!("PE1 answer: {}", answer);
        assert_eq!(answer, "233168");
    }
}
