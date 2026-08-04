# `ListPlot`

Plots a list of values as points.

```scrut
$ wo 'Head[ListPlot[{1, 2, 3, 4}]]'
Graphics
```

Datasets can be wrapped in `Labeled` to attach a label to each dataset:

```scrut
$ wo 'Head[ListPlot[{Labeled[Sqrt[Range[40]], "sqrt"], Labeled[Log[Range[40, 80]], "log"]}]]'
Graphics
```

Accepts the same core options as `Plot`, plus:

- **`Joined`** — `True` connects the points with a line.
- **`PlotMarkers`** — shape spec for the markers.

`Filling` accepts a rule list to fill between datasets: `{1 -> {2}}`
draws a stem from every point of dataset 1 to dataset 2, interpolating
linearly when the datasets are irregularly spaced:

```scrut
$ wo 'Head[ListPlot[{{1, 2, 3}, {2, 3, 4}}, Filling -> {1 -> {2}}]]'
Graphics
```

```scrut
$ wo 'Head[ListPlot[{Sort@First@Last[Reap[Plot[Sin[x], {x, 0, 2 Pi}, EvaluationMonitor :> Sow[{x, Sin[x]}]]]], Sort@First@Last[Reap[Plot[Cos[x], {x, 0, 2 Pi}, EvaluationMonitor :> Sow[{x, Cos[x]}]]]]}, Filling -> {1 -> {2}}]]'
Graphics
```

Values wrapped in `Around` are plotted at their central value
with error bars spanning the uncertainty:

```scrut
$ wo 'Head[ListPlot[{Around[2.2, 1.2], Around[3.3, 1.1], Around[5.9, 0.6]}]]'
Graphics
```

With a single series a `PlotStyle` list is one combined style rather than a
per-series cycle, so `{PointSize[0.04], Red}` gives one large red series;
with several series the list cycles instead, one style per series:

```scrut
$ wo 'StringContainsQ[ExportString[ListPlot[{{1, 1}, {2, 2}}, PlotStyle -> {PointSize[0.04], Red}], "SVG"], "<svg"]'
True
```

```scrut
$ wo 'StringContainsQ[ExportString[ListPlot[{{{1, 1}, {2, 2}}, {{1, 2}, {2, 3}}}, PlotStyle -> {Red, Green}], "SVG"], "<svg"]'
True
```

`PlotMarkers` draws a glyph at every data point instead of the round dot.
`Style` around the marker gives it a colour and a font size, and a list of
markers is cycled over the datasets:

```scrut
$ wo 'StringContainsQ[ExportString[ListPlot[{{1, 1}, {2, 2}, {3, 3}}, PlotMarkers -> Style["A", Red, 18]], "SVG"], "<svg"]'
True
```

```scrut
$ wo 'StringContainsQ[ExportString[ListPlot[{{{1, 1}, {2, 2}}, {{1, 2}, {2, 3}}}, PlotMarkers -> {"A", "B"}], "SVG"], "<svg"]'
True
```

`Epilog` primitives are drawn over the points, in data coordinates:

```scrut
$ wo 'StringContainsQ[ExportString[ListPlot[{{0, 0.}, {1, 1.}}, Epilog -> {Green, Line[{{0, 0.5}, {1, 0.5}}]}], "SVG"], "<svg"]'
True
```
