# `PieChart`

Pie chart of a list of values.

```scrut
$ wo 'Head[PieChart[{1, 2, 3}]]'
Graphics
```

`LabelingFunction -> f` labels every wedge with `f[value]`. Wrapping the
result in `Placed[label, position]` decides where along the wedge's radius
the label sits: `"RadialCenter"` (the default) centres it, `"RadialInner"`
and `"RadialOuter"` push it towards the hub or the rim, and
`"RadialCallout"` puts it outside the pie on a leader line.

```scrut
$ wo 'Head[PieChart[{0.25, 0.75}, LabelingFunction -> (Placed[Row[{NumberForm[100 #, 2], "%"}, " "], "RadialCallout"] &)]]'
Graphics
```

Options: `ChartLabels`, `ChartLegends`, `ChartStyle`, `ImageSize`,
`LabelingFunction`, `PlotLabel`.
