# `ShortestCurveDistance`

Gives the geodesic distance between two points on a region — the length of
the shortest curve found by `FindShortestCurve`.

```scrut
$ wo 'ShortestCurveDistance[Circle[], {1, 0}, {0, 1}]'
Pi/2
```

On a sphere it is the great-circle distance, kept symbolic for symbolic
coordinates:

```scrut
$ wo 'ShortestCurveDistance[Sphere[], {1, 0, 0}, {x, y, z}]'
ArcCos[x]
```

In a convex solid it reduces to the Euclidean distance:

```scrut
$ wo 'ShortestCurveDistance[Disk[], {0, 1}, {x, y}]'
Sqrt[x^2 + (-1 + y)^2]
```

Along a curve region it is the length of the path on the polyline:

```scrut
$ wo 'Round[ShortestCurveDistance[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], {1, 0}, {3, 0}], 10.^-6]'
2.828427
```
