use std::fs;

fn main() {
    let content = fs::read_to_string("apps/project-euler/data/triangle.txt")
        .expect("Failed to read triangle.txt");

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
    println!("The answer to Problem 67 is: {}", answer);
}
