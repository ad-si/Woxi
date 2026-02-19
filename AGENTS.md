## Scope

This repo contains several related sub-projects:

- Woxi
    - Interpreter for a subset of the Wolfram Language
    - Is a computer algebra system (CAS) and therefore all computations must be solved symbolically
- Woxi Playground
    - A simple playground to try out Woxi in the browser (tests/cli/playground)
- Woxi JupyterLite
    - JupyterLite instance with an integrated Woxi kernel (tests/cli/jupyterlite)


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
    The output of Woxi must 100% match the output of wolframscript.
- Do not write code to temporary files. Simply use `cargo run -- eval '<code>'`.
- If you stumble upon a pre-existing issue, do not try to ignore it or work around it,
    but rather fix it right away.
- Always execute scripts with a `timeout 20` to avoid getting stuck.
    If any of the scripts take a longer to execute, it should be considered a bug.
- After implementing a new function, make sure to update functions.csv
    including a description and an effect_level.
