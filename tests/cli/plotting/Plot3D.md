# `Plot3D`

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
