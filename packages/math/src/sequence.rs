use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Iterator that yields triangle numbers: 1, 3, 6, 10, 15, ...
pub struct TriangleNumbers {
    n: u64,
    current: u64,
}

impl TriangleNumbers {
    pub fn new() -> Self {
        Self { n: 0, current: 0 }
    }
}

impl Default for TriangleNumbers {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for TriangleNumbers {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.n += 1;
        self.current += self.n;
        Some(self.current)
    }
}

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

/// Find the length of a chain with caching to avoid redundant computation.
///
/// Like `find_chain_length`, but uses a cache to remember previously computed
/// chain lengths. When the same number appears in multiple chains, its length
/// is computed only once.
///
/// # Example
///
/// ```
/// use math::sequence::find_chain_length_cached;
/// use std::collections::HashMap;
///
/// let mut cache = HashMap::new();
/// let length = find_chain_length_cached(69u64, |n| {
///     let mut sum = 0;
///     let mut x = *n;
///     while x > 0 {
///         let d = x % 10;
///         sum += (1..=d).product::<u64>();
///         x /= 10;
///     }
///     sum
/// }, &mut cache);
/// assert_eq!(length, 5);
/// // The cache now contains lengths for all visited values
/// assert!(cache.contains_key(&69));
/// ```
pub fn find_chain_length_cached<T, F>(
    start: T,
    transform: F,
    cache: &mut HashMap<T, usize>,
) -> usize
where
    T: Eq + Hash + Clone,
    F: Fn(&T) -> T,
{
    let mut path: Vec<T> = Vec::new();
    let mut current = start;

    loop {
        // Check if already cached
        if let Some(&len) = cache.get(&current) {
            // Cache all elements in path
            for (i, elem) in path.iter().enumerate() {
                let path_len = path.len() - i + len;
                cache.insert(elem.clone(), path_len);
            }
            return path.len() + len;
        }

        // Check if current is in path (cycle detected)
        if let Some(cycle_start) = path.iter().position(|x| x == &current) {
            let cycle_len = path.len() - cycle_start;
            // Cache cycle elements (they all have the same length)
            for elem in path[cycle_start..].iter() {
                cache.insert(elem.clone(), cycle_len);
            }
            // Cache pre-cycle elements
            for (i, elem) in path[..cycle_start].iter().enumerate() {
                let path_len = path.len() - i;
                cache.insert(elem.clone(), path_len);
            }
            return path.len();
        }

        // Add current to path and continue
        path.push(current.clone());
        current = transform(&current);
    }
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

    #[test]
    fn test_find_chain_length_cached_simple_cycle() {
        let mut cache = HashMap::new();
        // 1 → 1 (immediate cycle)
        assert_eq!(find_chain_length_cached(1, |_| 1, &mut cache), 1);
        assert_eq!(cache.get(&1), Some(&1));
    }

    #[test]
    fn test_find_chain_length_cached_longer_chain() {
        let mut cache = HashMap::new();
        // 5 → 4 → 3 → 2 → 1 → 1 (cycle at 1)
        assert_eq!(
            find_chain_length_cached(5, |n| if *n > 1 { n - 1 } else { 1 }, &mut cache),
            5
        );
        // All values should be cached
        assert_eq!(cache.get(&5), Some(&5));
        assert_eq!(cache.get(&4), Some(&4));
        assert_eq!(cache.get(&3), Some(&3));
        assert_eq!(cache.get(&2), Some(&2));
        assert_eq!(cache.get(&1), Some(&1));
    }

    #[test]
    fn test_find_chain_length_cached_reuses_cache() {
        let mut cache = HashMap::new();
        // First call: 3 → 2 → 1 → 1
        assert_eq!(
            find_chain_length_cached(3, |n| if *n > 1 { n - 1 } else { 1 }, &mut cache),
            3
        );
        // Second call starting at 5: should use cached value for 3
        // 5 → 4 → 3 (cached: length 3) = 2 + 3 = 5
        assert_eq!(
            find_chain_length_cached(5, |n| if *n > 1 { n - 1 } else { 1 }, &mut cache),
            5
        );
    }

    #[test]
    fn test_find_chain_length_cached_cycle_not_at_start() {
        let mut cache = HashMap::new();
        // 0 → 1 → 2 → 3 → 2 (cycle between 2 and 3, but 0 and 1 are not in cycle)
        // Chain: 0, 1, 2, 3 = 4 unique values
        assert_eq!(
            find_chain_length_cached(0, |n| if *n < 3 { n + 1 } else { 2 }, &mut cache),
            4
        );
        // Verify cache: cycle elements (2, 3) have length 2, pre-cycle have longer
        assert_eq!(cache.get(&2), Some(&2));
        assert_eq!(cache.get(&3), Some(&2));
        assert_eq!(cache.get(&1), Some(&3));
        assert_eq!(cache.get(&0), Some(&4));
    }
}
