---
icon: lucide/shapes
---

# Geometry

Synthetic-geometry predicate tests on concrete numeric coordinates.

## `GeometricTest`

Tests whether one or more geometric objects satisfy a named property or
relation, returning `True` or `False`.

Points are collinear when they all lie on a single line.

```scrut
$ wo 'GeometricTest[{{2, 3}, {4, 6}, {-2, -3}}, "Collinear"]'
True
```

```scrut
$ wo 'GeometricTest[{{0, 0}, {1, 1}, {2, 3}}, "Collinear"]'
False
```

Polygon predicates such as `"Convex"` accept a `Polygon` object.

```scrut
$ wo 'GeometricTest[Polygon[{{0, 0}, {5, 1}, {4, 4}, {-2, 0}}], "Convex"]'
True
```

A square is convex, regular, and a rectangle.

```scrut
$ wo 'GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}], "Regular"]'
True
```

```scrut
$ wo 'GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}], "Rectangle"]'
True
```

Lines can be tested for `"Parallel"` and `"Perpendicular"`.

```scrut
$ wo 'GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], InfiniteLine[{{0, 1}, {1, 2}}]}, "Parallel"]'
True
```

```scrut
$ wo 'GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], InfiniteLine[{{0, 0}, {1, -1}}]}, "Perpendicular"]'
True
```

Triangles can be compared with `"Congruent"` and `"Similar"`.

```scrut
$ wo 'GeometricTest[{Triangle[{{0, 0}, {3, 0}, {0, 4}}], Triangle[{{0, 0}, {6, 0}, {0, 8}}]}, "Similar"]'
True
```

```scrut
$ wo 'GeometricTest[{Triangle[{{0, 0}, {3, 0}, {0, 4}}], Triangle[{{0, 0}, {6, 0}, {0, 8}}]}, "Congruent"]'
False
```

Multiple properties are tested simultaneously; the result is `True` only when
all of them hold.

```scrut
$ wo 'GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}], "Convex", "Rectangle"]'
True
```

## `ConvexHullMesh`

The convex hull of a 2D point set is returned as a `BoundaryMeshRegion`.
Interior and collinear-on-edge points are dropped; the remaining corners stay
in input order and the boundary `Line` walks them counter-clockwise.

```scrut
$ wo 'ConvexHullMesh[{{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}}]'
BoundaryMeshRegion[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, {Line[{{1, 2}, {2, 3}, {3, 4}, {4, 1}}]}, Method -> {SeparateBoundaries -> False}, WorkingPrecision -> Infinity]
```

A machine-real coordinate switches every vertex to a real and drops the
`WorkingPrecision` option.

```scrut
$ wo 'ConvexHullMesh[{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}]'
BoundaryMeshRegion[{{0., 0.}, {1., 0.}, {0., 1.}, {1., 1.}}, {Line[{{1, 2}, {2, 4}, {4, 3}, {3, 1}}]}, Method -> {SeparateBoundaries -> False}]
```
