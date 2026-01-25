use std::collections::HashMap;

fn cycle_length(d: u32) -> usize {
    let mut x = 1u32;
    let mut seen: HashMap<(u32, u32), usize> = HashMap::new();
    let mut pos = 0usize;

    loop {
        while x < d {
            x *= 10;
        }

        if x.is_multiple_of(d) {
            return 0; // no recurring cycle
        }

        let digit = x / d;
        x %= d;

        if let Some(&prev_pos) = seen.get(&(x, digit)) {
            return pos - prev_pos;
        }

        seen.insert((x, digit), pos);
        pos += 1;
    }
}

pub fn solve(verbose: bool) -> String {
    let mut best_d = 1;
    let mut best_cycle = 0;

    for d in 1..1000 {
        let cycle = cycle_length(d);
        if cycle > best_cycle {
            best_cycle = cycle;
            best_d = d;
        }
    }

    if verbose {
        println!("Best cycle length: {}", best_cycle);
    }

    best_d.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        let answer = solve(false);
        println!("PE26 answer: {}", answer);
        assert_eq!(answer, "983");
    }
}
