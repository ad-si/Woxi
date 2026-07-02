# `StackedListPlot`

Plots several datasets with their `y`-values accumulated, so each successive
dataset is stacked on top of the running total of the preceding ones. The
regions between consecutive cumulative curves are filled by default.

```scrut
$ wo 'Head[StackedListPlot[{{1, 2, 3, 4}, {2, 3, 4, 5}}]]'
Graphics
```

A single dataset stacks against the axis, like a filled line plot:

```scrut
$ wo 'Head[StackedListPlot[{1, 2, 3, 4}]]'
Graphics
```
