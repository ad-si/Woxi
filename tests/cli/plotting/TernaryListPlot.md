# `TernaryListPlot`

Plots triples `{a, b, c}` as points inside a ternary (barycentric) triangle.
Each triple is normalized so that `a + b + c == 1`; component 1 pulls the
point towards the top corner, component 2 towards the bottom-left corner and
component 3 towards the bottom-right corner.

```scrut
$ wo 'Head[TernaryListPlot[{{1, 1, 1}, {1, 2, 3}}]]'
Graphics
```

Multiple datasets can be plotted together by passing a list of lists of
triples, each drawn in its own color:

```scrut
$ wo 'Head[TernaryListPlot[{{{1, 1, 1}, {2, 1, 1}}, {{1, 2, 1}, {1, 1, 2}}}]]'
Graphics
```

Accepts the `ImageSize` and `PlotStyle` options.
