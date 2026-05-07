# `Plot`

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
