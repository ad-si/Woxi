# `ListPlot`

Plots a list of values as points.

```scrut
$ wo 'Head[ListPlot[{1, 2, 3, 4}]]'
Graphics
```

Accepts the same core options as `Plot`, plus:

- **`Joined`** — `True` connects the points with a line.
- **`PlotMarkers`** — shape spec for the markers.
