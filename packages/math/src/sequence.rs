use std::collections::HashSet;
use std::hash::Hash;

/// Find the length of a chain produced by repeatedly applying a transform function.
///
/// Starting from `start`, repeatedly applies `transform` until a previously seen
/// value is encountered. Returns the total number of unique values in the chain.
///
/// # Note
///
/// The cycle detected may not include the starting value. For example, if the
/// sequence is `1 → 2 → 3 → 4 → 3` (where 3 → 4 → 3 forms a cycle), the chain
/// length is 4 (values: 1, 2, 3, 4), but value 1 and 2 are not part of the cycle.
/// This function returns the total chain length, not just the cycle length.
///
/// # Example
///
/// ```
/// use math::sequence::find_chain_length;
///
/// // Collatz-like example: 69 → 363600 → 1454 → 169 → 363601 → 1454 (cycle)
/// // Chain length is 5 unique values before hitting a repeat
/// let length = find_chain_length(69, |n| {
///     let mut sum = 0;
///     let mut x = n;
///     while x > 0 {
///         let d = x % 10;
///         sum += (1..=d).product::<u64>();
///         x /= 10;
///     }
///     sum
/// });
/// assert_eq!(length, 5);
/// ```
pub fn find_chain_length<T, F>(start: T, transform: F) -> usize
where
    T: Eq + Hash + Clone,
    F: Fn(T) -> T,
{
    let mut seen = HashSet::new();
    let mut current = start;
    while !seen.contains(&current) {
        seen.insert(current.clone());
        current = transform(current);
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_chain_length_simple_cycle() {
        // 1 → 1 (immediate cycle)
        assert_eq!(find_chain_length(1, |_| 1), 1);
    }

    #[test]
    fn test_find_chain_length_longer_chain() {
        // 5 → 4 → 3 → 2 → 1 → 1 (cycle at 1)
        assert_eq!(find_chain_length(5, |n| if n > 1 { n - 1 } else { 1 }), 5);
    }

    #[test]
    fn test_find_chain_length_cycle_not_at_start() {
        // 0 → 1 → 2 → 3 → 2 (cycle between 2 and 3, but 0 and 1 are not in cycle)
        // Chain: 0, 1, 2, 3 = 4 unique values
        assert_eq!(find_chain_length(0, |n| if n < 3 { n + 1 } else { 2 }), 4);
    }
}
