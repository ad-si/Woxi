# `SolveValues`

Returns the values from solving equations, without the `Rule` wrappers
[`Solve`](Solve.md) uses.

The result follows the shape of the variable specification.
A bare symbol gives a flat list of values:

```scrut
$ wo 'SolveValues[x^2 == 4, x]'
{-2, 2}
```

A list of variables gives one value-list per solution —
even when the list holds a single variable:

```scrut
$ wo 'SolveValues[x^2 == 4, {x}]'
{{-2}, {2}}
```

```scrut
$ wo 'SolveValues[{x + y == 3, x - y == 1}, {x, y}]'
{{2, 1}}
```
