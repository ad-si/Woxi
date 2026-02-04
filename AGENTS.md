## Scope

- Woxi is a re-implementation of a subset of the Wolfram Language
- Woxi is a computer algebra system and therefore all computations must be solved symbolically


## Development

- Whenever you change something, add tests for it to one of the `tests/*.rs` files.
- When fixing a bug, always add a regression test so the bug wont't occur again.
- Always run `make test` after any changes to rule out any regressions.
- Never implement features or tests only for special cases like e.g. one specific number.
    Make sure to implement it for all possible cases!
- Use `wolframscript -code 'Plus[1, 2]'` to verify the output of Wolfram Language code.
