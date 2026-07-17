# `PolygonCoordinates`

Returns the vertices of a 2-D polygon in canonical (sorted) order.

```scrut
$ wo 'PolygonCoordinates[Polygon[{{2, 2}, {0, 0}, {1, 1}, {3, 0}}]]'
{{0, 0}, {1, 1}, {2, 2}, {3, 0}}
```

It also accepts the `Triangle` head.

```scrut
$ wo 'PolygonCoordinates[Triangle[{{0, 0}, {2, 0}, {0, 2}}]]'
{{0, 0}, {0, 2}, {2, 0}}
```

Degenerate (zero-area) polygons stay unevaluated.

```scrut
$ wo 'PolygonCoordinates[Polygon[{{0, 0}, {1, 0}, {2, 0}}]]'
PolygonCoordinates[Polygon[{{0, 0}, {1, 0}, {2, 0}}]]
```
