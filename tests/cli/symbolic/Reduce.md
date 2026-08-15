# `Reduce`

Simplifies a logical condition, e.g. a polynomial equation, to an
equivalent form describing all solutions.

```scrut
$ wo 'Reduce[x^2 == 4, x]'
x == -2 || x == 2
```

Woxi keeps specialized built-in paths for polynomial equations, integer
intervals, complex algebra, and common transcendental forms. On native builds,
exact polynomial constraints over the reals can additionally use SMT-RAT's
CAlC quantifier elimination when the `smtrat-shared` executable is available.
Set `WOXI_SMTRAT` to a different executable path when necessary.

`WOXI_REDUCE_BACKEND` controls selection:

- `auto` (the default) keeps fast built-in reductions and invokes SMT-RAT for
  quantified formulas or exact real formulas left unresolved by Woxi.
- `internal` disables subprocess use. WebAssembly builds use this behavior in
  `auto` mode because they cannot start native subprocesses.
- `smtrat` requires SMT-RAT for supported exact-real formulas and reports a
  backend error if it cannot run.

The subprocess timeout defaults to 30 seconds and can be changed with
`WOXI_SMTRAT_TIMEOUT_MS`.
