# `Reduce`

Simplifies a logical condition, e.g. a polynomial equation, to an
equivalent form describing all solutions.

```scrut
$ wo 'Reduce[x^2 == 4, x]'
x == -2 || x == 2
```

Exact linear formulas with an explicit `Reals`, `Rationals` or `Integers`
domain are decided by exact quantifier elimination (Fourier-Motzkin over the
reals and rationals, Cooper's algorithm over the integers), so nested `Exists`
and `ForAll` are eliminated:

```scrut
$ wo 'Reduce[Exists[y, x < y && y < 1], x, Reals]'
x < 1
```

```scrut
$ wo 'Reduce[Exists[y, x == 2 y + 1] && 0 <= x <= 6, x, Integers]'
x == 1 || x == 3 || x == 5
```

Equations between the requested variables are solved:

```scrut
$ wo 'Reduce[x + y == 3 && x - y == 1, {x, y}, Integers]'
x == 2 && y == 1
```

The linear fragment accepts exact rational affine terms; `==`, `!=`, `<`,
`<=`, `>` and `>=`; `And`, `Or`, `Not` and `Xor`; nested `Exists` and
`ForAll`; and, over the integers, `Divisible` and `Mod[affine, n] == r`.
Nonlinear, approximate, algebraic-number and transcendental inputs keep
Woxi's specialized routes.
