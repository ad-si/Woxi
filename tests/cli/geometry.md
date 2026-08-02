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

## `RegionProduct`

The Cartesian product of regions, which is answered with a named primitive
rather than a product object. Coordinates of the earlier arguments come
first, so two segments span a rectangle:

```scrut
$ wo 'RegionProduct[Line[{{0}, {2}}], Line[{{0}, {3}}]]'
Rectangle[{0, 0}, {2, 3}]
```

A disk swept along a segment is a cylinder, a triangle swept along one a
prism:

```scrut
$ wo 'RegionProduct[Disk[{1, 2}, 3], Interval[{4, 6}]]'
Cylinder[{{1, 2, 4}, {1, 2, 6}}, 3]
```

```scrut
$ wo 'RegionProduct[Triangle[], Interval[{0, 2}]]'
Prism[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 2}, {1, 0, 2}, {0, 1, 2}}]
```

The product is measured like any other region:

```scrut
$ wo 'RegionMeasure[RegionProduct[Disk[], Line[{{0}, {1}}]]]'
Pi
```

A pair with no primitive to become is left standing:

```scrut
$ wo 'RegionProduct[Disk[], Disk[]]'
RegionProduct[Disk[{0, 0}], Disk[{0, 0}]]
```

## `CircularArcThrough`

The arc of the circle through the given points, written as a `Circle` running
from the smallest of their angles about the centre to the largest.

```scrut
$ wo 'CircularArcThrough[{{1, 0}, {0, 1}, {-1, 0}}]'
Circle[{0, 0}, 1, {0, Pi}]
```

Two points alone are the ends of a diameter:

```scrut
$ wo 'CircularArcThrough[{{1, 0}, {0, 1}}]'
Circle[{1/2, 1/2}, 1/Sqrt[2], {(3*Pi)/4, (7*Pi)/4}]
```

The result is a region like any other arc:

```scrut
$ wo 'ArcLength[CircularArcThrough[{{1, 0}, {0, 1}, {-1, 0}}]]'
Pi
```

Points no circle passes through — collinear ones, or too few — are refused:

```scrut
$ wo 'CircularArcThrough[{{0, 0}, {1, 0}, {2, 0}}]'

CircularArcThrough::indep: CircularArcThrough does not exist for {{0, 0}, {1, 0}, {2, 0}}.
CircularArcThrough[{{0, 0}, {1, 0}, {2, 0}}]
```

## `ConvexPolyhedronQ`

Whether a region is a convex polyhedron: bounded, flat-faced and
three-dimensional.

```scrut
$ wo 'ConvexPolyhedronQ[Cube[]]'
True
```

A curved solid is not one, and neither is a flat region:

```scrut
$ wo 'ConvexPolyhedronQ[Ball[]]'
False
```

```scrut
$ wo 'ConvexPolyhedronQ[Simplex[2]]'
False
```

Corners of your own are measured — skewing one so a side stops being flat
takes the convexity with it:

```scrut
$ wo 'ConvexPolyhedronQ[Prism[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}]]'
True
```

```scrut
$ wo 'ConvexPolyhedronQ[Prism[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {5, 0, 1}, {0, 1, 1}}]]'
False
```

## `PolyhedronData`

Named polyhedra and their properties: the Platonic solids, plus the
Archimedean solids with icosahedral symmetry and their Catalan duals.

```scrut
$ wo 'PolyhedronData["TruncatedIcosahedron", "FaceCount"]'
32
```

The football: twelve pentagons and twenty hexagons.

```scrut
$ wo 'Tally[Length /@ PolyhedronData["TruncatedIcosahedron", "FaceIndices"]]'
{{5, 12}, {6, 20}}
```

Metric properties stay exact:

```scrut
$ wo 'PolyhedronData["Icosidodecahedron", "Volume"]'
(45 + 17*Sqrt[5])/6
```

An Archimedean solid has a circumsphere but no insphere, its dual the
other way round:

```scrut
$ wo 'PolyhedronData["RhombicTriacontahedron", "Circumradius"]'
Missing[NotApplicable]
```

`"Faces"` gives the corners and the faces that index into them, in one
`GraphicsComplex`:

```scrut
$ wo 'Head[PolyhedronData["Icosahedron", "Faces"]]'
GraphicsComplex
```
