## Scope

- Woxi is a re-implementation of a subset of the Wolfram Language
- Woxi is a computer algebra system and therefore all computations must be solved symbolically


## Development

There is 3 levels of tests:

1. Unit tests in ./tests/*.rs files.
    They are the fastest to execute and
    there should be a unit test for every feature / language construct here.
2. Snapshots tests in ./tests/scripts/*.wls files
    They ensure the code also works in files and
    has exactly the same output when run with `wolframscript`.
3. Documentation tests in ./tests/cli/*.md files.
    Serves as documentation and also ensures `woxi` and `wolframscript`
    have the same output for shorter and more diverse code snippets.
    Those tests should be less comprehensive than the unit tests and
    focus on illustrative examples.

**Follow those rules:**

- Whenever you change something, add unit tests for it.
- When fixing a bug, always add a regression test so the bug wont't occur again.
- Always run `make test` after any changes to rule out any regressions.
- Never implement features or tests only for special cases like e.g. one specific number.
    Make sure to implement it for all possible cases!
- Use `wolframscript -code 'Plus[1, 2]'` to verify the output of Wolfram Language code.
- Do not write code to temporary files. Simply use `cargo run -- eval '<code>'`.
