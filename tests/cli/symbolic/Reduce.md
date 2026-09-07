# `Reduce`

Simplifies a logical condition, e.g. a polynomial equation, to an
equivalent form describing all solutions.

```scrut
$ wo 'Reduce[x^2 == 4, x]'
x == -2 || x == 2
```

Woxi keeps specialized built-in paths for polynomial equations, integer
intervals, complex algebra, and common transcendental forms. Exact linear
formulas over the reals and rationals use Woxi's self-contained exact
Fourier-Motzkin engine, including nested quantifiers:

```scrut
$ wo 'Reduce[Exists[y, x < y && y < 1], x, Reals]'
x < 1
```

Exact linear integer formulas use Woxi's self-contained Presburger engine.
Unbounded solution sets are represented symbolically rather than searched up
to an arbitrary cap:

```scrut
$ wo 'Reduce[Exists[y, x == 2 y + 1], x, Integers]'
Element[x, Integers] && Mod[x, 2] == 1
```

Neither engine invokes Wolfram, an SMT solver, or another subprocess. Inputs
outside the documented exact-linear fragment continue to use Woxi's
specialized internal fallback routes.

The systematic fragment accepts exact rational affine terms (constants,
variables, constant multiples, sums, and differences); `==`, `!=`, `<`, `<=`,
`>`, and `>=`; arbitrary `And`, `Or`, `Not`, and `Xor`; and nested `Exists` and
`ForAll`. Integer formulas additionally accept `Divisible` and
`Mod[affine, positiveInteger] == residue` (or its negation). Target domains are
explicit `Reals`, `Rationals`, or `Integers`. Nonlinear products/powers,
approximate coefficients, mixed integer/real theories, algebraic-number
coefficients, and transcendental atoms are outside this systematic fragment;
some retain separate specialized Woxi behavior.
