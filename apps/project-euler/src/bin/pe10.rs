use math::primes_below;

fn main() {
    let primes = primes_below(2_000_000);
    let sum: u64 = primes.iter().map(|&p| p as u64).sum();
    println!("The answer to Problem 10 is: {}", sum);
}
