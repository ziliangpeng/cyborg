pub mod solutions;

/// Available problem IDs
pub const PROBLEMS: &[u32] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 26, 30, 34, 42, 50, 67, 74, 81,
];

/// Run a specific problem and return the answer
/// If verbose is true, the solution may print debug information
pub fn run_problem(id: u32, verbose: bool) -> Option<String> {
    match id {
        1 => Some(solutions::pe1::solve(verbose)),
        2 => Some(solutions::pe2::solve(verbose)),
        3 => Some(solutions::pe3::solve(verbose)),
        4 => Some(solutions::pe4::solve(verbose)),
        5 => Some(solutions::pe5::solve(verbose)),
        6 => Some(solutions::pe6::solve(verbose)),
        7 => Some(solutions::pe7::solve(verbose)),
        8 => Some(solutions::pe8::solve(verbose)),
        9 => Some(solutions::pe9::solve(verbose)),
        10 => Some(solutions::pe10::solve(verbose)),
        11 => Some(solutions::pe11::solve(verbose)),
        13 => Some(solutions::pe13::solve(verbose)),
        14 => Some(solutions::pe14::solve(verbose)),
        16 => Some(solutions::pe16::solve(verbose)),
        26 => Some(solutions::pe26::solve(verbose)),
        30 => Some(solutions::pe30::solve(verbose)),
        34 => Some(solutions::pe34::solve(verbose)),
        42 => Some(solutions::pe42::solve(verbose)),
        50 => Some(solutions::pe50::solve(verbose)),
        67 => Some(solutions::pe67::solve(verbose)),
        74 => Some(solutions::pe74::solve(verbose)),
        81 => Some(solutions::pe81::solve(verbose)),
        _ => None,
    }
}
