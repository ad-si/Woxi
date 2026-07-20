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
