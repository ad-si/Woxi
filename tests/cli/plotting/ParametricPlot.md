# `ParametricPlot`

Plots a 2D parametric curve.

```scrut
$ wo 'Head[ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]]'
Graphics
```

Accepts the same options as `Plot`.

Several curves are given as a list of `{fx, fy}` pairs, nested to any depth —
the grouping only steers styling:

```scrut
$ wo 'Head[ParametricPlot[{{{Cos[t], Sin[t]}, {2 Cos[t], 2 Sin[t]}}}, {t, 0, 2 Pi}]]'
Graphics
```

The curve specification is held, so one built by `Table` needs no `Evaluate`:

```scrut
$ wo 'Head[ParametricPlot[Table[{a Cos[t], a Sin[t]}, {a, 1, 3}], {t, 0, 2 Pi}]]'
Graphics
```
