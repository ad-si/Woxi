# `Graphics3D`

3D analogue of `Graphics`. Primitives include `Cuboid`, `Sphere`,
`Cylinder`, `Cone`, `Line`, `Polygon`, `Tube`, `Point`.

```scrut
$ wo 'Head[Graphics3D[{Red, Cuboid[]}]]'
Graphics3D
```

Options: `ImageSize`, `PlotRange`, `BoxRatios`, `Boxed`, `Axes`,
`AxesLabel`, `Lighting`, `ViewPoint`, `ViewVertical`, `Background`,
`SphericalRegion`.

`SphericalRegion -> True` scales the picture so the sphere enclosing the
contents fits the display area, which keeps the scale fixed as the view
turns or the contents move:

```scrut
$ wo 'Head[Graphics3D[Sphere[], SphericalRegion -> True]]'
Graphics3D
```

`PlotLabel -> label` sets a title above the picture, the same way it does
for a 2-D `Graphics`. Any expression can be the label; it is typeset
rather than printed, so the drawn label reads `Torus Knot`:

```scrut
$ wo 'StringContainsQ[ExportString[Graphics3D[Sphere[], PlotLabel -> Style[Row[{"Torus ", "Knot"}], FontSize -> 18]], "SVG"], "<svg"]'
True
```

`BSplineCurve[{p1, p2, …}]` draws the B-spline its control points define,
and `Tube[BSplineCurve[…], r]` runs a tube of radius `r` along that same
curve — how a knot is given its thickness:

```scrut
$ wo 'Head[Graphics3D[BSplineCurve[{{0, 0, 0}, {1, 2, 0}, {2, 0, 1}}]]]'
Graphics3D
```

The tube is a solid, so it is drawn as a surface rather than as the bare
curve:

```scrut
$ wo 'StringContainsQ[ExportString[Graphics3D[Tube[BSplineCurve[{{0, 0, 0}, {1, 2, 0}, {2, 0, 1}}], 0.2]], "SVG"], "<svg"]'
True
```

The unbounded primitives `InfiniteLine`, `HalfLine`, `InfinitePlane` and
`HalfPlane` draw the part of themselves that lies inside the picture's
box, so an infinite line clipped to `PlotRange -> 10` is the same drawing
as the segment between its two exit points:

```scrut
$ wo 'Head[Graphics3D[InfiniteLine[{{0, 0, 0}, {1, 1, 1}}], PlotRange -> 10]]'
Graphics3D
```

`Sphere[{p1, p2, …}, r]` is a whole set of spheres of radius `r`, one per
centre — how a scene marks several points at once, and the same drawing as
one `Sphere` per centre:

```scrut
$ wo 'Head[Graphics3D[Sphere[{{1, 0, 0}, {-1, 0, 0}}, 0.5]]]'
Graphics3D
```
