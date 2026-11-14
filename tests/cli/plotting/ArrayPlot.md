# `ArrayPlot`

Plots a matrix as a grid of colored cells.

```scrut
$ wo 'Head[ArrayPlot[{{0, 1}, {1, 0}}]]'
Graphics
```

The plot sits in a frame, so `FrameLabel` labels its edges. Its axes are
rows and columns, so the labels are written as `{rows, columns}` (left, then
bottom) or `{{bottom, top}, {left, right}}`:

```scrut
$ wo 'Head[ArrayPlot[{{0, 1}, {1, 0}}, FrameLabel -> {{"", "top"}, {"", ""}}]]'
Graphics
```

An explicit `PlotRange` sets the values the colors span: white at its
minimum, black at its maximum. Cells outside it show the background unless
`ClippingStyle` gives them a color:

```scrut
$ wo 'Head[ArrayPlot[{{-1, 1}, {1, 3}}, PlotRange -> {0, 2}, ClippingStyle -> Red]]'
Graphics
```
