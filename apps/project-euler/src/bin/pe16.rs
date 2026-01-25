use math::BigInt;

fn main() {
    let mut n = BigInt::from_u32(2);

    for _ in 0..999 {
        n = n * 2;
    }

    let s = format!("{}", n);
    let digit_sum: u32 = s.chars().map(|c| c.to_digit(10).unwrap()).sum();

    println!("The answer to Problem 16 is: {}", digit_sum);
}
