# `Graphics`

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
