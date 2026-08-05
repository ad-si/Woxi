# `Plot`

Plots a function (or list of functions) over a range.

```scrut
$ wo 'Head[Plot[Sin[x], {x, 0, Pi}]]'
Graphics
```

### Options

- **`PlotRange`** — y-axis range as `{ymin, ymax}` or full
  `{{xmin, xmax}, {ymin, ymax}}`; also `All` or `Automatic`.
  `Automatic` leaves out extreme outliers so a pole can't flatten the rest
  of the curve; `All` shows every sampled value.
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
- **`Filling`** — region to fill (`None`, `Axis`, `Bottom`, `Top`, or a
  rule list like `{1 -> {2}}` to fill between curves).
- **`EvaluationMonitor`** — expression evaluated at every sampled point.
- **`FillingStyle`** — style directive for the filled region.
- **`Mesh`** — controls mesh markers (`None`, `All`, an integer).
- **`PlotTheme`** — named theme like `"Scientific"` or `"Business"`.
- **`ColorFunction`** — named color map (e.g. `"Rainbow"`).
- **`Background`** — background color.
- **`Frame`** — draw a frame around the plot (`True` / `False`).
- **`FrameLabel`** — labels for the frame sides.
- **`Epilog`** — extra graphics drawn on top of the plot.
- **`Prolog`** — extra graphics drawn beneath the plot.

`EvaluationMonitor` can record the points Plot samples, e.g. via `Sow`:

```scrut
$ wo 'Positive[Length[First@Last[Reap[Plot[Sin[x], {x, 0, 2 Pi}, EvaluationMonitor :> Sow[x]]]]]]'
True
```

Tick labels follow the Wolfram Language's rules: a value outside
`[10^-5, 10^6)` is written in scientific notation (`6×10¹⁵`, not
`6000000000000000`), and inside that range the labels stay plain with
enough decimals to tell neighbouring ticks apart (`0.0002`). Either way
the plot exports:

```scrut
$ wo 'StringContainsQ[ExportString[Plot[x, {x, 0, 6*10^15}], "SVG"], "<svg"]'
True
```

```scrut
$ wo 'StringContainsQ[ExportString[Plot[x, {x, 0, 0.001}], "SVG"], "<svg"]'
True
```

`PlotRange -> All` keeps a steep curve inside the frame — here the top
tick reaches `E^10`, which the automatic range trims away:

```scrut
$ wo 'StringContainsQ[ExportString[Plot[E^x, {x, 0, 10}, PlotRange -> All], "SVG"], "<svg"]'
True
```

A tick given as a bare position is labelled with that expression rather
than with its decimal expansion, so an axis divided into multiples of π
reads back as multiples of π:

```scrut
$ wo 'StringContainsQ[ExportString[Plot[Sin[x], {x, 0, 2 Pi}, Ticks -> {{0, Pi/2, Pi, 3 Pi/2, 2 Pi}, Automatic}], "SVG"], "<svg"]'
True
```

`Filling` rules may be grouped in sub-lists — `{{1 -> {2}}, {3 -> 0}}`
shades the same regions as the flat `{1 -> {2}, 3 -> 0}`:

```scrut
$ wo 'StringContainsQ[ExportString[Plot[{Sin[x], 2 Sin[x], 3 Sin[x]}, {x, 0, 2 Pi}, Filling -> {{1 -> {2}}, {3 -> 0}}], "SVG"], "<svg"]'
True
```
