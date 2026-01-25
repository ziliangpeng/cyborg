use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

fn word_value(word: &str) -> u32 {
    word.chars().map(|c| (c as u32) - ('A' as u32) + 1).sum()
}

pub fn solve(verbose: bool) -> String {
    let data_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data", "words.txt"]
        .iter()
        .collect();
    let content = fs::read_to_string(&data_path).expect("Failed to read words.txt");

    // Parse words and compute their values
    let word_values: Vec<u32> = content
        .split(',')
        .map(|w| w.trim_matches('"'))
        .map(|w| word_value(w))
        .collect();

    let max_value = *word_values.iter().max().unwrap();

    // Generate triangle numbers up to max_value
    let mut triangle_numbers: HashSet<u32> = HashSet::new();
    let mut n = 1u32;
    loop {
        let t = n * (n + 1) / 2;
        if t > max_value {
            break;
        }
        triangle_numbers.insert(t);
        n += 1;
    }

    if verbose {
        println!("Max word value: {}", max_value);
        println!("Triangle numbers up to max: {:?}", triangle_numbers.len());
    }

    let count = word_values
        .iter()
        .filter(|v| triangle_numbers.contains(v))
        .count();

    count.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_value() {
        // SKY = 19 + 11 + 25 = 55
        assert_eq!(word_value("SKY"), 55);
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "162");
    }
}
