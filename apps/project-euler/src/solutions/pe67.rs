use std::fs;
use std::path::PathBuf;

pub fn solve(verbose: bool) -> String {
    let _ = verbose;
    let data_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data", "triangle.txt"]
        .iter()
        .collect();
    let content = fs::read_to_string(&data_path).expect("Failed to read triangle.txt");

    let elem: Vec<Vec<u32>> = content
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        })
        .collect();

    let n = elem.len();
    let mut sum: Vec<Vec<u32>> = vec![vec![0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let left_up = if j > 0 && i > 0 { sum[i - 1][j - 1] } else { 0 };
            let right_up = if j < i && i > 0 { sum[i - 1][j] } else { 0 };
            sum[i][j] = left_up.max(right_up) + elem[i][j];
        }
    }

    let answer = sum[n - 1].iter().max().unwrap();
    answer.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "7273");
    }
}
