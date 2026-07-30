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
- **`FrameTicks`** — `False` or `None` keeps the border but drops the tick
  marks and their labels.

An option may be written with `:>`, which holds its right-hand side until
the option is used:

```scrut
$ wo 't = 1100; StringContainsQ[ExportString[Graphics[{Disk[]}, PlotLabel :> Which[t == 1100, "met", True, ""]], "SVG"], ">met<"]'
True
```

```scrut
$ wo 'StringContainsQ[ExportString[Graphics[{Line[{{0, 0}, {1, 1}}]}, Frame -> True, FrameTicks -> False], "SVG"], "<text"]'
False
```
