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

With two parameters it draws the image of the parameter rectangle.
`ColorFunction` colors it, `Mesh` and `MeshFunctions` draw level curves
of functions of `x`, `y` and the two parameters over it, and
`BoundaryStyle` styles its outline:

```scrut
$ wo 'Head[ParametricPlot[{u Cos[v], u Sin[v]}, {u, 1, 2}, {v, 0, Pi}, ColorFunction -> (Hue[#4] &), Mesh -> {5, 3}, MeshFunctions -> {#1 + #2 &, #3 &}, BoundaryStyle -> None]]'
Graphics
```
