use clap::Parser;
use project_euler::{PROBLEMS, run_problem};

#[derive(Parser)]
#[command(name = "project-euler")]
#[command(about = "Run Project Euler solutions")]
struct Cli {
    /// Run a specific problem by number
    #[arg(short, long)]
    problem: Option<u32>,

    /// Run all available problems
    #[arg(short, long)]
    all: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.all {
        // Run all problems with verbose=false for cleaner output
        for &id in PROBLEMS {
            if let Some(answer) = run_problem(id, false) {
                println!("The answer to Problem {} is: {}", id, answer);
            }
        }
    } else if let Some(id) = cli.problem {
        // Run single problem with verbose=true for debug info
        match run_problem(id, true) {
            Some(answer) => println!("The answer to Problem {} is: {}", id, answer),
            None => eprintln!("Problem {} not implemented", id),
        }
    } else {
        eprintln!("Usage: project-euler --problem <N> or --all");
        eprintln!("Available problems: {:?}", PROBLEMS);
    }
}
