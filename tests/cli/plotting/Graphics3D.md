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
