# `NSolve`

Numeric equation solver.

```scrut
$ wo 'NSolve[x^2 - 2 == 0, x]'
{{x -> -1.4142135623730951}, {x -> 1.414213562373095}}
```

A bare polynomial stands for the equation `poly == 0`:

```scrut
$ wo 'NSolve[x^2 - 2, x]'
{{x -> -1.4142135623730951}, {x -> 1.414213562373095}}
```

A cubic with machine-real coefficients has one real root and a conjugate
pair:

```scrut
$ wo 'Round[x /. NSolve[x^3 + 1.5 x^2 - 3.2 x + 4.7 == 0, x], 1/10^6]'
{-19079/6250, 2426/3125 - (120997*I)/125000, 2426/3125 + (120997*I)/125000}
```

A repeated root is listed once per multiplicity:

```scrut
$ wo 'NSolve[x^3 - 4. x^2 == 0, x]'
{{x -> 0.}, {x -> 0.}, {x -> 4.}}
```

The variable may be left out, as with `Solve`:

```scrut
$ wo 'NSolve[x^2 == 2]'
\{\{x -> -1\.414213562373095\d?\}, \{x -> 1\.414213562373095\d?\}\} (regex)
```

Constraints narrow the answer to the root inside them:

```scrut
$ wo 'NSolve[1 == f^2 (3 - f) && 0 <= f <= 1]'
{{f -> 0.6527036446661393}}
```
