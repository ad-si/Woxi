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

Values wrapped in `Around` are plotted at their central value
with error bars spanning the uncertainty:

```scrut
$ wo 'Head[ListPlot[{Around[2.2, 1.2], Around[3.3, 1.1], Around[5.9, 0.6]}]]'
Graphics
```
