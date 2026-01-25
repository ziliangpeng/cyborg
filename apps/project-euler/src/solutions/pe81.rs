pub fn solve(_verbose: bool) -> String {
    let input = include_str!("../../data/matrix.txt");
    let data: Vec<Vec<u64>> = input
        .lines()
        .map(|line| line.split(',').map(|n| n.parse().unwrap()).collect())
        .collect();

    let rows = data.len();
    let cols = data[0].len();
    let mut sum = vec![vec![0u64; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            let from_top = if i > 0 { sum[i - 1][j] } else { 0 };
            let from_left = if j > 0 { sum[i][j - 1] } else { 0 };

            sum[i][j] = if i == 0 && j == 0 {
                data[i][j]
            } else if i == 0 {
                from_left + data[i][j]
            } else if j == 0 {
                from_top + data[i][j]
            } else {
                from_top.min(from_left) + data[i][j]
            };
        }
    }

    sum[rows - 1][cols - 1].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "427337");
    }
}
