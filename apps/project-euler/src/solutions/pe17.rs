use std::collections::HashMap;

fn letter_count(n: u32, ones: &HashMap<u32, &str>, tens: &HashMap<u32, &str>) -> usize {
    if n >= 1 && n <= 19 {
        return ones.get(&n).unwrap().len();
    }

    if n >= 20 && n <= 99 {
        let tens_digit = n / 10;
        let ones_digit = n % 10;
        let mut count = tens.get(&tens_digit).unwrap().len();
        if ones_digit > 0 {
            count += ones.get(&ones_digit).unwrap().len();
        }
        return count;
    }

    if n >= 100 && n <= 999 {
        let hundreds_digit = n / 100;
        let remainder = n % 100;
        let mut count = ones.get(&hundreds_digit).unwrap().len() + "hundred".len();
        if remainder > 0 {
            count += "and".len();
            count += letter_count(remainder, ones, tens);
        }
        return count;
    }

    if n == 1000 {
        return "onethousand".len();
    }

    0
}

pub fn solve(verbose: bool) -> String {
    let _ = verbose;

    let mut ones: HashMap<u32, &str> = HashMap::new();
    ones.insert(1, "one");
    ones.insert(2, "two");
    ones.insert(3, "three");
    ones.insert(4, "four");
    ones.insert(5, "five");
    ones.insert(6, "six");
    ones.insert(7, "seven");
    ones.insert(8, "eight");
    ones.insert(9, "nine");
    ones.insert(10, "ten");
    ones.insert(11, "eleven");
    ones.insert(12, "twelve");
    ones.insert(13, "thirteen");
    ones.insert(14, "fourteen");
    ones.insert(15, "fifteen");
    ones.insert(16, "sixteen");
    ones.insert(17, "seventeen");
    ones.insert(18, "eighteen");
    ones.insert(19, "nineteen");

    let mut tens: HashMap<u32, &str> = HashMap::new();
    tens.insert(2, "twenty");
    tens.insert(3, "thirty");
    tens.insert(4, "forty");
    tens.insert(5, "fifty");
    tens.insert(6, "sixty");
    tens.insert(7, "seventy");
    tens.insert(8, "eighty");
    tens.insert(9, "ninety");

    let total: usize = (1..=1000).map(|n| letter_count(n, &ones, &tens)).sum();
    total.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dicts() -> (HashMap<u32, &'static str>, HashMap<u32, &'static str>) {
        let mut ones: HashMap<u32, &str> = HashMap::new();
        ones.insert(1, "one");
        ones.insert(2, "two");
        ones.insert(3, "three");
        ones.insert(4, "four");
        ones.insert(5, "five");
        ones.insert(6, "six");
        ones.insert(7, "seven");
        ones.insert(8, "eight");
        ones.insert(9, "nine");
        ones.insert(10, "ten");
        ones.insert(11, "eleven");
        ones.insert(12, "twelve");
        ones.insert(13, "thirteen");
        ones.insert(14, "fourteen");
        ones.insert(15, "fifteen");
        ones.insert(16, "sixteen");
        ones.insert(17, "seventeen");
        ones.insert(18, "eighteen");
        ones.insert(19, "nineteen");

        let mut tens: HashMap<u32, &str> = HashMap::new();
        tens.insert(2, "twenty");
        tens.insert(3, "thirty");
        tens.insert(4, "forty");
        tens.insert(5, "fifty");
        tens.insert(6, "sixty");
        tens.insert(7, "seventy");
        tens.insert(8, "eighty");
        tens.insert(9, "ninety");

        (ones, tens)
    }

    #[test]
    fn test_letter_count_1_to_19() {
        let (ones, tens) = make_dicts();
        assert_eq!(letter_count(1, &ones, &tens), 3); // "one"
        assert_eq!(letter_count(2, &ones, &tens), 3); // "two"
        assert_eq!(letter_count(3, &ones, &tens), 5); // "three"
        assert_eq!(letter_count(5, &ones, &tens), 4); // "five"
    }

    #[test]
    fn test_letter_count_20_to_99() {
        let (ones, tens) = make_dicts();
        assert_eq!(letter_count(20, &ones, &tens), 6); // "twenty"
        assert_eq!(letter_count(22, &ones, &tens), 9); // "twentytwo"
        assert_eq!(letter_count(47, &ones, &tens), 10); // "fortyseven"
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "21124");
    }
}
