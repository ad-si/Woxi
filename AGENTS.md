## Scope

- Woxi is a re-implementation of a subset of the Wolfram Language
- Woxi is a computer algebra system and therefore all computations must be solved symbolically


## Development

- Always run `make test` after any changes
- Never implement features or tests only for special cases like e.g. one specific number.
    Make sure to implement it for all possible cases!
- Use `wolframscript -code 'Plus[1, 2]'` to verify the output of Wolfram Language code.
- When fixing a bug, always add a regression test so the bug wont't occur again.
