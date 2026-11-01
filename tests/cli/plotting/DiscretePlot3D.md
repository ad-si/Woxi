# `DiscretePlot3D`

3D plot of a function at discrete grid points.

Each value is drawn as a point with a stem down to the bottom of the plot;
`ExtentSize -> Full` draws bars one sampling step wide instead.

```scrut
$ wo 'Count[DiscretePlot3D[i + j, {i, 0, 2}, {j, 0, 2}, ExtentSize -> Full], _Cuboid, Infinity]'
9
```
