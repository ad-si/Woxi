# `ArrayPlot`

Plots a matrix as a grid of colored cells.

```scrut
$ wo 'Head[ArrayPlot[{{0, 1}, {1, 0}}]]'
Graphics
```

The plot sits in a frame, so `FrameLabel` labels its edges — written as
`{{left, right}, {bottom, top}}`:

```scrut
$ wo 'Head[ArrayPlot[{{0, 1}, {1, 0}}, FrameLabel -> {{"", ""}, {"", "top"}}]]'
Graphics
```
