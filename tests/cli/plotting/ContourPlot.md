# `ContourPlot`

Draws contours of a 2-variable function.

```scrut
$ wo 'Head[ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]]'
Graphics
```

```scrut
$ wo 'Head[ContourPlot[Sin[x] Sin[y], {x, -3, 3}, {y, -3, 3}, Contours -> {-0.5, 0, 0.5}, ContourShading -> False]]'
Graphics
```

The contours live in a `GraphicsComplex`, the first element of the
primitive list that is the first part of the result:

```scrut
$ wo 'Head /@ First[ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]]'
{GraphicsComplex, List}
```

### Options

In addition to the common options above, `ContourPlot` recognizes:

- **`Contours`** — an integer (number of contours) or a list of levels.
- **`ContourStyle`** — directives applied to the contour lines.
- **`ContourShading`** — `True` / `False` / `Automatic`.
- **`ColorFunction`** — color map for the shaded regions.
- **`PlotPoints`** / **`MaxRecursion`** — sampling control.
