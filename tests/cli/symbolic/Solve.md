# `Solve`

Symbolic equation solver.

```scrut
$ wo 'Solve[x^2 == 4, x]'
{{x -> -2}, {x -> 2}}
```

```scrut
$ wo 'Solve[{x + y == 3, x - y == 1}, {x, y}]'
{{x -> 2, y -> 1}}
```

A negative leading coefficient still gives simplified, correctly ordered roots.

```scrut
$ wo 'Solve[2 - x^2 == 0, x]'
{{x -> -Sqrt[2]}, {x -> Sqrt[2]}}
```

```scrut
$ wo 'Solve[6 - x - x^2 == 0, x]'
{{x -> -3}, {x -> 2}}
```

`x^n == c` is solved in radicals, the roots being the real root times the
`n`-th roots of unity:

```scrut
$ wo 'Solve[x^3 == 8, x]'
{{x -> 2}, {x -> -2*(-1)^(1/3)}, {x -> 2*(-1)^(2/3)}}
```

```scrut
$ wo 'Solve[x^3 == 2, x]'
{{x -> -(-2)^(1/3)}, {x -> 2^(1/3)}, {x -> (-1)^(2/3)*2^(1/3)}}
```

For an odd degree and a negative right-hand side the generating root is the
real one:

```scrut
$ wo 'Solve[x^3 == -8, x]'
{{x -> -2}, {x -> 2*(-1)^(1/3)}, {x -> -2*(-1)^(2/3)}}
```

`NSolve` stays on the numeric root finder, so a conjugate pair agrees to the
last bit:

```scrut
$ wo 'NSolve[x^3 == 8, x]'
{{x -> -1. - 1.7320508075688772*I}, {x -> -1. + 1.7320508075688772*I}, {x -> 2.}}
```
