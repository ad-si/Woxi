# `ParametricPlot3D`

3D parametric plot.

```scrut
$ wo 'Head[ParametricPlot3D[{Cos[t], Sin[t], t}, {t, 0, 6}]]'
Graphics3D
```

A list of triples draws one curve per triple. The items only have to *take*
triple shape once evaluated, so a solution can be substituted into each of
them — how a trajectory is drawn beside its two coordinate projections:

```scrut
$ wo "s = NDSolve[{x'[t] == -x[t], y'[t] == x[t] - y[t], x[0] == 1, y[0] == 0}, {x[t], y[t]}, {t, 0, 3}]; Head[ParametricPlot3D[{{0, x[t], y[t]} /. s, {t, x[t], 0} /. s, {t, 2, y[t]} /. s}, {t, 0, 3}]]"
Graphics3D
```

`PlotStyle -> Tube[r]` draws the curve as a tube of radius `r` rather than
as a line; any colour given alongside it still applies:

```scrut
$ wo 'Head[ParametricPlot3D[{Cos[t], Sin[t], t/5}, {t, 0, 6.2}, PlotStyle -> {Red, Tube[0.2]}]]'
Graphics3D
```
