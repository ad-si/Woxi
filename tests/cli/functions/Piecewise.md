# `Piecewise`

Defines a piecewise-defined expression from a list of
`{value, condition}` pairs.

```scrut
$ wo 'Piecewise[{{1, x > 0}}] /. x -> 1'
1
```

The pieces are held: the value of a piece whose condition is `False` is never
evaluated, which is what makes the construct usable as a guard.  Here `1/x` is
not divided by zero:

```scrut
$ wo 'Piecewise[{{1/x, x != 0}}, 0] /. x -> 0'
0
```

The same guard keeps a tabulated curve inside its own data range, so querying
outside it emits no extrapolation warning:

```scrut
$ wo 'f = Interpolation[{{0, 0}, {1, 2}, {2, 0}}, InterpolationOrder -> 1]; g[w_] := Piecewise[{{f[w], 0 <= w <= 2}}, 0]; {g[1], g[5]}'
{2, 0}
```
