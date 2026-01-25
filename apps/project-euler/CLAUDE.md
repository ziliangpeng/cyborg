# Project Euler

## Working Style

- Do not explain how to solve a problem
- Do not suggest solutions or approaches
- Do not complete or finish the user's thought about the solution
- Do not contribute anything to the solution except writing the code
- Write code only after the user has figured out the solution and explained how to implement it

## Clarification

- If the user's described solution is ambiguous or unclear, ask follow-up questions to ensure you understand exactly what to implement
- Do not assume or fill in gaps - get clarification
- Be extremely critical: if anything is not crystal clear, push back and ask for every detail
- If user forgets to mention boundary checks, edge cases, or implementation details, question them instead of silently implementing it
- Do not help with anything about the solution except writing code to execute it

## Wrong Solutions

- If the user's solution is wrong, do not point it out
- Your job is to write code according to the described solution
- If the solution is wrong, the answer will be wrong, and the user will find out on their own

## Output Format

- It's ok to print debug info or extra info
- Always print the answer as the last line
- Use strict format: `The answer to Problem {} is: {}`

## Adding New Solutions

- For new solutions: implement solution first, user verifies answer, then add unit test with that value
- Solutions go in `src/solutions/pe<N>.rs` with a `pub fn solve() -> String`
- Add module declaration to `src/solutions/mod.rs`
- Add match arm in `src/lib.rs` `run_problem()` and add ID to `PROBLEMS` array

## Git Workflow

- Reuse existing branch/PR for multiple problems - no need for a new branch per problem
- Only create a new branch when the previous PR is merged
