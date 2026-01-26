fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        0 => 31, // January
        1 => {
            // February - check leap year
            if year % 4 != 0 {
                28
            } else if year % 100 == 0 && year % 400 != 0 {
                28
            } else {
                29
            }
        }
        2 => 31,  // March
        3 => 30,  // April
        4 => 31,  // May
        5 => 30,  // June
        6 => 31,  // July
        7 => 31,  // August
        8 => 30,  // September
        9 => 31,  // October
        10 => 30, // November
        11 => 31, // December
        _ => panic!("Invalid month"),
    }
}

pub fn solve(verbose: bool) -> String {
    let _ = verbose;

    let mut day = 1; // 1 Jan 1900 is Monday (0 = Sunday, 1 = Monday, ...)

    // Advance through 1900 to find what day 1 Jan 1901 is
    for month in 0..12 {
        day = (day + days_in_month(1900, month)) % 7;
    }

    if verbose {
        println!("1 Jan 1901 is day {}", day);
    }

    // Count Sundays from 1901 to 2000
    let mut count = 0;
    for year in 1901..=2000 {
        for month in 0..12 {
            if day == 0 {
                count += 1;
            }
            day = (day + days_in_month(year, month)) % 7;
        }
    }

    count.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2000, 1), 29); // 2000 is leap (divisible by 400)
        assert_eq!(days_in_month(1900, 1), 28); // 1900 is not leap (divisible by 100 but not 400)
        assert_eq!(days_in_month(2004, 1), 29); // 2004 is leap
        assert_eq!(days_in_month(2001, 1), 28); // 2001 is not leap
    }

    #[test]
    fn test_solve() {
        assert_eq!(solve(false), "171");
    }
}
