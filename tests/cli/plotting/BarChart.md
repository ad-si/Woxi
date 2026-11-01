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

`ChartStyle -> "Pastel"` colors each bar with a distinct color from the
named gradient color scheme.

```scrut
$ wo 'Head[BarChart[{1, 2, 3, 4}, ChartStyle -> "Pastel"]]'
Graphics
```

The values of an association are the bar heights; `PlotLabels -> Automatic`
(the current name of `ChartLabels`) labels the bars with the keys.

```scrut
$ wo 'Head[BarChart[<|"a" -> 1, "b" -> 2, "c" -> 5|>, PlotLabels -> Automatic]]'
Graphics
```

### Options

- **`BarOrigin`** — `Bottom` (default, vertical) or `Left`/`Right` (horizontal).
- **`LabelingFunction`** — applied to each bar value to label the bar's end.

- **`ChartLabels`** — category labels along the axis.
- **`ChartLegends`** — legend entries.
- **`ChartStyle`** — color / directive, list of directives, or a named
  `ColorData` gradient scheme (each bar gets a distinct color sampled from
  the scheme). Supported scheme names: `"Pastel"`, `"Rainbow"`,
  `"SolarColors"`, `"TemperatureMap"`, `"ThermometerColors"`,
  `"DarkRainbow"`, `"Aquamarine"`, `"StarryNightColors"`, `"AvocadoColors"`,
  `"SunsetColors"`, `"FruitPunchColors"`, `"CherryTones"`.
- **`ChartElementFunction`** — custom shape for each bar.
- **`BarSpacing`** — gap between bars.
- **`ImageSize`**, **`PlotLabel`**, **`AxesLabel`**, **`Frame`**,
  **`FrameLabel`**, **`PlotRange`** — as for `Plot`.
