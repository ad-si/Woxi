# `BarChart`

Bar chart of a list of values.

```scrut
$ wo 'Head[BarChart[{1, 2, 3, 4}]]'
Graphics
```

`BarOrigin -> Left` (or `Right`) renders a horizontal bar chart, with
categories down the left edge and bars growing sideways from zero.

```scrut
$ wo 'Head[BarChart[{1, 2, 3, 4}, BarOrigin -> Left]]'
Graphics
```

### Options

- **`BarOrigin`** — `Bottom` (default, vertical) or `Left`/`Right` (horizontal).
- **`LabelingFunction`** — applied to each bar value to label the bar's end.

- **`ChartLabels`** — category labels along the axis.
- **`ChartLegends`** — legend entries.
- **`ChartStyle`** — color / directive or list of directives.
- **`ChartElementFunction`** — custom shape for each bar.
- **`BarSpacing`** — gap between bars.
- **`ImageSize`**, **`PlotLabel`**, **`AxesLabel`**, **`Frame`**,
  **`FrameLabel`**, **`PlotRange`** — as for `Plot`.
