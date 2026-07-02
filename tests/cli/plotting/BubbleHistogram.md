# `BubbleHistogram`

Bubble histogram — bins two-dimensional data into a rectangular grid and
draws a bubble at the center of every non-empty bin, its size encoding the
number of points that fell in it.

```scrut
$ wo 'Head[BubbleHistogram[{{1, 2}, {1, 2}, {3, 4}, {5, 1}}]]'
Graphics
```

An optional second argument controls the binning, mirroring `Histogram`:
an integer sets the number of bins per axis, and `{dx}` sets the bin width.

```scrut
$ wo 'Head[BubbleHistogram[{{1, 1}, {2, 2}, {3, 3}, {4, 4}}, 2]]'
Graphics
```

Options: `AxesLabel`, `PlotLabel`, `ChartStyle`, `ImageSize`.
