pub mod solutions;

/// Available problem IDs
pub const PROBLEMS: &[u32] = &[1, 10, 13, 14, 16, 26, 50, 67];

/// Run a specific problem and return the answer
/// If verbose is true, the solution may print debug information
pub fn run_problem(id: u32, verbose: bool) -> Option<String> {
    match id {
        1 => Some(solutions::pe1::solve(verbose)),
        10 => Some(solutions::pe10::solve(verbose)),
        13 => Some(solutions::pe13::solve(verbose)),
        14 => Some(solutions::pe14::solve(verbose)),
        16 => Some(solutions::pe16::solve(verbose)),
        26 => Some(solutions::pe26::solve(verbose)),
        50 => Some(solutions::pe50::solve(verbose)),
        67 => Some(solutions::pe67::solve(verbose)),
        _ => None,
    }
}
