use std::collections::HashMap;

fn transform(n: u64) -> u64 {
    if n % 2 == 0 {
        n / 2
    } else {
        3 * n + 1
    }
}

fn main() {
    const LIMIT: u64 = 1_000_000;
    let mut memo: HashMap<u64, u64> = HashMap::new();
    memo.insert(1, 1);

    for start in 2..LIMIT {
        let mut chain = vec![start];
        let mut n = start;

        // Follow the chain until we hit a memoized number
        while !memo.contains_key(&n) {
            n = transform(n);
            chain.push(n);
        }

        // Now n is memoized, update all numbers in the chain
        let base_length = memo[&n];
        for (i, &num) in chain.iter().enumerate() {
            let length = (chain.len() - i - 1) as u64 + base_length;
            memo.insert(num, length);
        }
    }

    // Find the number with longest path in range 1..LIMIT
    let mut max_start = 1;
    let mut max_length = 1;

    for n in 1..LIMIT {
        if let Some(&length) = memo.get(&n) {
            if length > max_length {
                max_length = length;
                max_start = n;
            }
        }
    }

    println!("The answer to Problem 14 is: {}", max_start);
}
