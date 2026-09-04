# `GeometricTransformation`

Draw graphics mapped through an affine transformation.

The second argument may be a `TransformationFunction`, which is normalized to
the `{matrix, vector}` pair of the affine map `p ↦ matrix.p + vector`.

```scrut
$ wo 'GeometricTransformation[Line[{{0, 0}, {2, 0}}], ReflectionTransform[{-1, 1}]]'
GeometricTransformation[Line[{{0, 0}, {2, 0}}], {{{0, 1}, {1, 0}}, {0, 0}}]
```

Reflecting in the line `y = x` swaps the coordinates of every point, which is
how the graph of a function's inverse is drawn from the graph of the function.

```scrut
$ wo 'ReflectionTransform[{-1, 1}][{2, 0}]'
{0, 2}
```

Inside a `Graphics`, the enclosed primitives are drawn through that map — the
horizontal line above comes out vertical:

```scrut
$ wo 'StringContainsQ[ExportString[Graphics[GeometricTransformation[Line[{{0, 0}, {2, 0}}], ReflectionTransform[{-1, 1}]], PlotRange -> {{-1, 3}, {-1, 3}}], "SVG"], "<svg"]'
True
```
