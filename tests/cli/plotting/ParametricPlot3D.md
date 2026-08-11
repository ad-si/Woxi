# `ParametricPlot3D`

3D parametric plot.

```scrut
$ wo 'Head[ParametricPlot3D[{Cos[t], Sin[t], t}, {t, 0, 6}]]'
Graphics3D
```

`PlotStyle -> Tube[r]` draws the curve as a tube of radius `r` rather than
as a line; any colour given alongside it still applies:

```scrut
$ wo 'StringContainsQ[ExportString[ParametricPlot3D[{Cos[t], Sin[t], t/5}, {t, 0, 6.2}, PlotStyle -> {Red, Tube[0.2]}], "SVG"], "<polygon"]'
True
```
