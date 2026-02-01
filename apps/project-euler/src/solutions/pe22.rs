use std::fs;
use std::path::PathBuf;

pub fn solve(verbose: bool) -> String {
    let data_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data"].iter().collect();
    let data_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "data", "names.txt"]
        .iter()
        .collect();

    if !data_path.exists() {
        fs::create_dir_all(&data_dir).expect("Failed to create data directory");
        let response = ureq::get("https://projecteuler.net/resources/documents/0022_names.txt")
            .call()
            .expect("Failed to download names.txt");
        let content = response
            .into_string()
            .expect("Failed to read names.txt response");
        fs::write(&data_path, content).expect("Failed to write names.txt");
    }

    let content = fs::read_to_string(&data_path).expect("Failed to read names.txt");

    let mut names: Vec<&str> = content.split(',').map(|s| s.trim_matches('"')).collect();

    names.sort();

    let mut total: u64 = 0;

    for (i, name) in names.iter().enumerate() {
        let position = (i + 1) as u64;
        let alpha_value: u64 = name.chars().map(|c| (c as u64) - ('A' as u64) + 1).sum();
        let score = alpha_value * position;

        if verbose {
            println!(
                "{}: {} -> alpha={}, score={}",
                position, name, alpha_value, score
            );
        }

        total += score;
    }

    total.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "871198282");
    }
}
