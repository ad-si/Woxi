# `LayeredGraphPlot`

Draws a graph in layers: every vertex sits one layer further from a root
than the parent that first reached it.

```scrut
$ wo 'Head[LayeredGraphPlot[{1 -> 2, 2 -> 3, 2 -> 4}]]'
Graphics
```

The second argument says which edge of the plot the roots go on, and the
layers grow away from it — `Left` draws the graph left to right:

```scrut
$ wo 'Head[LayeredGraphPlot[{1 -> 2, 2 -> 3, 2 -> 4}, Left]]'
Graphics
```

`DirectedEdges -> False` draws the edges as plain lines instead of
arrows, and `ImageSize` sizes the picture:

```scrut
$ wo 'Head[LayeredGraphPlot[{1 -> 2, 2 -> 3}, Left, DirectedEdges -> False, ImageSize -> {200, 50}]]'
Graphics
```
