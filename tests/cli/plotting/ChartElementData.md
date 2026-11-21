# `ChartElementData`

Lists the built-in `ChartElementFunction` names a chart function accepts.

```scrut
$ wo 'ChartElementData["DistributionChart"]'
{Density, DensityQuantile, FadingQuantile, GlassQuantile, HistogramDensity, LineDensity, PointDensity, Quantile, SmoothDensity}
```

`"HistogramDensity"` draws each dataset's histogram turned on its side
and mirrored about the dataset's position.

```scrut
$ wo 'Length[Cases[DistributionChart[{{1, 2, 2, 3}}, ChartElementFunction -> "HistogramDensity"], _Rectangle, Infinity]]'
3
```
