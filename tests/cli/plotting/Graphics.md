# `Graphics`

Wraps a list of 2D graphics primitives into a renderable object.
Typical primitives: `Line`, `Circle`, `Disk`, `Rectangle`, `Polygon`,
`Point`, `Text`, `Arrow`. Directives include `Red`, `Blue`, `Dashed`,
`Thickness[r]`, `PointSize[r]`, `Opacity[α]`, `RGBColor[r,g,b]`.

```scrut
$ wo 'Head[Graphics[{Red, Disk[]}]]'
Graphics
```

`Sphere` and `Ball` are drawn here too — in the plane a sphere is the circle
bounding it and a ball the filled disk. That is how the circumcircle of
three points reaches a picture, since `Circumsphere` returns a `Sphere`
whatever the dimension:

```scrut
$ wo 'Circumsphere[{{0, 0}, {4, 0}, {0, 3}}]'
Sphere[{2, 3/2}, 5/2]
```

```scrut
$ wo 'StringCount[ExportString[Graphics[{Circumsphere[{{0, 0}, {4, 0}, {0, 3}}]}], "SVG"], "<ellipse"]'
1
```

A `Style` around a label paints its `Background` behind the text, which is
how a label stays readable over whatever it is placed on:

```scrut
$ wo 'StringContainsQ[ExportString[Graphics[{Line[{{0, 0}, {4, 4}}], Style[Text["8", {2, 2}], Background -> White]}], "SVG"], "<rect"]'
True
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
