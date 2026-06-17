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
