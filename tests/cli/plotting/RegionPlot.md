# `RegionPlot`

Plots the region where an inequality holds.

```scrut
$ wo 'Head[RegionPlot[x^2 + y^2 < 1, {x, -1, 1}, {y, -1, 1}]]'
Graphics
```

A list of conditions plots one region per condition,
each in its own color.
All `Graphics` options apply.

```scrut
$ wo 'Head[RegionPlot[{y > x/2 + 1, y < 3 x/2 - 1}, {x, -5, 5}, {y, -5, 5}, GridLines -> Automatic, Axes -> True, AxesOrigin -> {0, 0}, AxesStyle -> Thick, Frame -> False]]'
Graphics
```
