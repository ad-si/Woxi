# Plotting

Woxi implements the most common 2D and 3D plotting functions from
the Wolfram Language, along with charting and graphics primitives.

Because graphical output cannot be compared textually, the examples
below use `Head[...]` to verify that a plot expression returns a
`Graphics` or `Graphics3D` value. In a Jupyter notebook or the
[playground](playground/index.html), the same expressions render as
SVG images.


# 2D Function Plots

## `Plot`

Plots a function (or list of functions) over a range.

```scrut
$ wo 'Head[Plot[Sin[x], {x, 0, Pi}]]'
Graphics
```

### Options

- **`PlotRange`** — y-axis range as `{ymin, ymax}` or full
  `{{xmin, xmax}, {ymin, ymax}}`; also `All` or `Automatic`.
- **`PlotStyle`** — one directive (`Red`, `Dashed`, `Thickness[0.01]`)
  or a list applied per curve.
- **`AxesLabel`** — `{xLabel, yLabel}`.
- **`PlotLabel`** — string/expression drawn above the plot.
- **`PlotLegends`** — legend labels.
- **`ImageSize`** — size in pixels, either a scalar or `{w, h}`.
- **`AspectRatio`** — height ÷ width ratio of the plot area.
- **`GridLines`** — `None`, `Automatic`, or `{xSpecs, ySpecs}`.
- **`Ticks`** — tick spec, `None`, or `Automatic`.
- **`Axes`** — show axes (`True`/`False` or `{xBool, yBool}`).
- **`AxesOrigin`** — location of the axes intersection.
- **`PlotPoints`** — initial sample count.
- **`MaxRecursion`** — sub-division depth for adaptive sampling.
- **`Filling`** — region to fill (`None`, `Axis`, `Bottom`, `Top`).
- **`FillingStyle`** — style directive for the filled region.
- **`Mesh`** — controls mesh markers (`None`, `All`, an integer).
- **`PlotTheme`** — named theme like `"Scientific"` or `"Business"`.
- **`ColorFunction`** — named color map (e.g. `"Rainbow"`).
- **`Background`** — background color.
- **`Frame`** — draw a frame around the plot (`True` / `False`).
- **`FrameLabel`** — labels for the frame sides.
- **`Epilog`** — extra graphics drawn on top of the plot.
- **`Prolog`** — extra graphics drawn beneath the plot.


## `ParametricPlot`

Plots a 2D parametric curve.

```scrut
$ wo 'Head[ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]]'
Graphics
```

Accepts the same options as `Plot`.


## `PolarPlot`

Plots a function in polar coordinates.

```scrut
$ wo 'Head[PolarPlot[1, {t, 0, 2 Pi}]]'
Graphics
```

Accepts the same options as `Plot`.


## `ContourPlot`

Draws contours of a 2-variable function.

```scrut
$ wo 'Head[ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]]'
Graphics
```

### Options

In addition to the common options above, `ContourPlot` recognizes:

- **`Contours`** — an integer (number of contours) or a list of levels.
- **`ContourStyle`** — directives applied to the contour lines.
- **`ContourShading`** — `True` / `False` / `Automatic`.
- **`ColorFunction`** — color map for the shaded regions.
- **`PlotPoints`** / **`MaxRecursion`** — sampling control.


## `DensityPlot`

Density plot of a 2-variable function.

```scrut
$ wo 'Head[DensityPlot[x + y, {x, 0, 1}, {y, 0, 1}]]'
Graphics
```

Same options as `ContourPlot`, plus `ColorFunctionScaling`.


## `RegionPlot`

Plots the region where an inequality holds.

```scrut
$ wo 'Head[RegionPlot[x^2 + y^2 < 1, {x, -1, 1}, {y, -1, 1}]]'
Graphics
```


# List plots

## `ListPlot`

Plots a list of values as points.

```scrut
$ wo 'Head[ListPlot[{1, 2, 3, 4}]]'
Graphics
```

Accepts the same core options as `Plot`, plus:

- **`Joined`** — `True` connects the points with a line.
- **`PlotMarkers`** — shape spec for the markers.


## `ListLinePlot`

Plots a list of values as a connected line (like `ListPlot` with
`Joined -> True`).

```scrut
$ wo 'Head[ListLinePlot[{1, 2, 3, 4}]]'
Graphics
```


# 3D Plots

## `Plot3D`

Plots a surface `z = f[x, y]`.

```scrut
$ wo 'Head[Plot3D[Sin[x + y], {x, 0, 2}, {y, 0, 2}]]'
Graphics3D
```

### Options

- **`PlotRange`** — `{zmin, zmax}` or `{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}`.
- **`PlotStyle`**, **`PlotLabel`**, **`PlotLegends`**,
  **`ImageSize`**, **`Background`**, **`PlotTheme`** — as for `Plot`.
- **`AxesLabel`** — `{xLabel, yLabel, zLabel}`.
- **`BoxRatios`** — relative side lengths of the enclosing box.
- **`Boxed`** — draw the box frame (`True`/`False`).
- **`ColorFunction`** — e.g. `"Rainbow"`, `"TemperatureMap"`.
- **`Mesh`** — `None`, `All`, an integer, or a list of specs.
- **`MeshFunctions`** — functions defining the mesh lines.
- **`MeshStyle`** / **`MeshShading`** — styling of the mesh.
- **`PlotPoints`** / **`MaxRecursion`** — sampling density.
- **`ViewPoint`** — viewing direction.
- **`ViewVertical`** — which axis is vertical.
- **`Lighting`** — light-source specification.
- **`Ticks`** / **`AxesEdge`** — axis control.


## `ListPointPlot3D`

3D scatter plot of a list of points.

```scrut
$ wo 'Head[ListPointPlot3D[{{1, 2, 3}, {4, 5, 6}}]]'
Graphics3D
```


# Charts

## `BarChart`

Bar chart of a list of values.

```scrut
$ wo 'Head[BarChart[{1, 2, 3, 4}]]'
Graphics
```

### Options

- **`ChartLabels`** — category labels along the axis.
- **`ChartLegends`** — legend entries.
- **`ChartStyle`** — color / directive or list of directives.
- **`ChartElementFunction`** — custom shape for each bar.
- **`BarSpacing`** — gap between bars.
- **`ImageSize`**, **`PlotLabel`**, **`AxesLabel`**, **`Frame`**,
  **`FrameLabel`**, **`PlotRange`** — as for `Plot`.


## `Histogram`

Histogram of a list of numeric samples.

```scrut
$ wo 'Head[Histogram[{1, 2, 2, 3, 3, 3}]]'
Graphics
```

Same options as `BarChart`, plus:

- **`Bins`** — `Automatic`, an integer (number of bins),
  or a list of bin edges.


## `PieChart`

Pie chart of a list of values.

```scrut
$ wo 'Head[PieChart[{1, 2, 3}]]'
Graphics
```

Options: `ChartLabels`, `ChartLegends`, `ChartStyle`, `ImageSize`,
`PlotLabel`.


## `BubbleChart`

Bubble chart — each point is drawn as a disk whose size encodes a
third coordinate.

```scrut
$ wo 'Head[BubbleChart[{{1, 2, 3}, {4, 5, 6}}]]'
Graphics
```

Options: `ChartLabels`, `ChartLegends`, `ChartStyle`, `BubbleSizes`,
`ImageSize`.


# Raw graphics

## `Graphics`

Wraps a list of 2D graphics primitives into a renderable object.
Typical primitives: `Line`, `Circle`, `Disk`, `Rectangle`, `Polygon`,
`Point`, `Text`, `Arrow`. Directives include `Red`, `Blue`, `Dashed`,
`Thickness[r]`, `PointSize[r]`, `Opacity[α]`, `RGBColor[r,g,b]`.

```scrut
$ wo 'Head[Graphics[{Red, Disk[]}]]'
Graphics
```

### Options

- **`ImageSize`**, **`PlotRange`**, **`AspectRatio`**, **`Axes`**,
  **`Frame`**, **`FrameLabel`**, **`Background`**, **`PlotLabel`**,
  **`Epilog`**, **`Prolog`** — as for `Plot`.


## `Graphics3D`

3D analogue of `Graphics`. Primitives include `Cuboid`, `Sphere`,
`Cylinder`, `Cone`, `Line`, `Polygon`, `Tube`, `Point`.

```scrut
$ wo 'Head[Graphics3D[{Red, Cuboid[]}]]'
Graphics3D
```

Options: `ImageSize`, `PlotRange`, `BoxRatios`, `Boxed`, `Axes`,
`AxesLabel`, `Lighting`, `ViewPoint`, `ViewVertical`, `Background`.


## `Show`

Combines multiple `Graphics` / `Graphics3D` objects into a single
plot.

```scrut
$ wo 'Head[Show[Plot[Sin[x], {x, 0, Pi}]]]'
Graphics
```
