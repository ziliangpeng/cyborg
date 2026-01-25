use std::collections::HashMap;

fn sum(i: usize, j: usize, memo: &mut HashMap<(usize, usize), u64>) -> u64 {
    // Swap if j > i (exploit symmetry)
    let (i, j) = if j > i { (j, i) } else { (i, j) };

    // Base case
    if i == 20 && j == 20 {
        return 1;
    }

    // Check memo
    if let Some(&val) = memo.get(&(i, j)) {
        return val;
    }

    // Compute: dp[i][j] = dp[i+1][j] + dp[i][j+1]
    let down = if i + 1 > 20 { 0 } else { sum(i + 1, j, memo) };
    let right = if j + 1 > 20 { 0 } else { sum(i, j + 1, memo) };
    let result = down + right;

    memo.insert((i, j), result);
    result
}

pub fn solve(verbose: bool) -> String {
    let _ = verbose;
    let mut memo = HashMap::new();
    sum(0, 0, &mut memo).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "137846528820");
    }
}
