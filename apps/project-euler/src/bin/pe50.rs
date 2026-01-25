use math::{is_prime, primes_below};

fn main() {
    const LIMIT: u64 = 1_000_000;
    let primes = primes_below(LIMIT as u32);

    let mut best_answer = 0u64;
    let mut best_chain_len = 0usize;
    let mut best_start_idx = 0usize;
    let mut best_end_idx = 0usize;

    for start_idx in 0..primes.len() {
        let mut sum = 0u64;
        for (count, &p) in primes[start_idx..].iter().enumerate() {
            sum += p as u64;
            if sum >= LIMIT {
                break;
            }
            let chain_len = count + 1;
            if chain_len > best_chain_len && is_prime(sum) {
                best_chain_len = chain_len;
                best_answer = sum;
                best_start_idx = start_idx;
                best_end_idx = start_idx + count;
            }
        }
    }

    println!("Chain length: {}", best_chain_len);
    println!("Start: {} (index {})", primes[best_start_idx], best_start_idx);
    println!("End: {} (index {})", primes[best_end_idx], best_end_idx);
    println!("The answer to Problem 50 is: {}", best_answer);
}
