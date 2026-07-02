# `FindShortestCurve`

Finds the shortest curve (minimizing geodesic) between two points on a
region. On a circle this is the shorter arc, returned as a `Circle` arc
with exact angles.

```scrut
$ wo 'FindShortestCurve[Circle[], {1, 0}, {0, 1}]'
Circle[{0, 0}, 1, {0, Pi/2}]
```

`ArcLength` of the result is the geodesic distance:

```scrut
$ wo 'ArcLength[FindShortestCurve[Circle[], {1, 0}, {0, 1}]]'
Pi/2
```

In a convex solid the shortest curve is the straight segment between the
points:

```scrut
$ wo 'FindShortestCurve[Disk[], {1/10, 4/5}, {-1/2, 0}]'
Line[{{1/10, 4/5}, {-1/2, 0}}]
```

Curve regions are treated as meshes: the result is the machine-precision
path along the polyline.

```scrut
$ wo 'FindShortestCurve[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], {1, 0}, {3, 0}]'
Line[{{1., 0.}, {2., 1.}, {3., 0.}}]
```
