// PE67 is a more complex version of this problem with a larger triangle.

const TRIANGLE: &str = "
75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
";

pub fn solve(verbose: bool) -> String {
    let _ = verbose;

    let elem: Vec<Vec<u32>> = TRIANGLE
        .trim()
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
        let answer = solve(false);
        println!("PE18 answer: {}", answer);
    }
}
