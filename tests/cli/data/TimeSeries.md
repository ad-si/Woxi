# `TimeSeries`

Represents a series of values paired with time stamps. Descriptive
statistics such as `Mean` and `Total` operate on the value path.

A time series can be given as a list of `{time, value}` pairs:

```scrut
$ wo 'Mean[TimeSeries[{{1, 10.}, {2, 20.}, {3, 30.}}]]'
20.
```

```scrut
$ wo 'Total[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]'
60
```

A bare list of values gets integer time stamps `1, 2, 3, …`:

```scrut
$ wo 'Mean[TimeSeries[{10., 20., 30., 40.}]]'
25.
```

`TimeSeriesWindow` keeps the points whose time stamps fall in a window,
including both endpoints:

```scrut
$ wo 'TimeSeriesWindow[TimeSeries[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}], {2, 3}]["Path"]'
{{2, 20}, {3, 30}}
```

Either bound may be infinite to leave that end open:

```scrut
$ wo 'TimeSeriesWindow[TimeSeries[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}], {3, Infinity}]["Values"]'
{30, 40}
```

`TimeSeriesResample` samples the piecewise-linear path at an even step,
interpolating exactly:

```scrut
$ wo 'TimeSeriesResample[TimeSeries[{{1, 10}, {3, 30}, {4, 40}}], 1]["Path"]'
{{1, 10}, {2, 20}, {3, 30}, {4, 40}}
```

```scrut
$ wo 'TimeSeriesResample[TimeSeries[{{1, 10}, {2, 15}, {4, 20}}], 1]["Path"]'
{{1, 10}, {2, 15}, {3, 35/2}, {4, 20}}
```

`TimeSeriesShift` moves every time stamp, and `TimeSeriesMap` transforms every
value:

```scrut
$ wo 'TimeSeriesShift[TimeSeries[{{1, 10}, {2, 20}}], 5]["Path"]'
{{6, 10}, {7, 20}}
```

```scrut
$ wo 'TimeSeriesMap[# + 1 &, TimeSeries[{{1, 10}, {2, 20}}]]["Path"]'
{{1, 11}, {2, 21}}
```

`TimeSeriesThread` hands the function the values the series share at each time
stamp:

```scrut
$ wo 'TimeSeriesThread[Total, {TimeSeries[{{1, 10}, {2, 20}}], TimeSeries[{{1, 1}, {2, 2}}]}]["Path"]'
{{1, 11}, {2, 22}}
```

`RegularlySampledQ` asks whether the stamps are evenly spaced, and
`TimeSeriesInsert` keeps the path sorted:

```scrut
$ wo '{RegularlySampledQ[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]], RegularlySampledQ[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}]]}'
{True, False}
```

```scrut
$ wo 'TimeSeriesInsert[TimeSeries[{{1, 10}, {3, 30}}], {2, 20}]["Path"]'
{{1, 10}, {2, 20}, {3, 30}}
```

Arithmetic keeps the time stamps and works on the values:

```scrut
$ wo 'Normal[2*TimeSeries[{{1, 10}, {2, 20}}] + 1]'
{{1, 21}, {2, 41}}
```

`TimeSeriesRescale` carries the time stamps linearly onto a given span,
keeping their spacing and the values:

```scrut
$ wo 'TimeSeriesRescale[TimeSeries[{{1, 10}, {2, 20}, {4, 40}}], {0, 1}]["Times"]'
{0, 1/3, 1}
```

A series runs in time order however its points were written:

```scrut
$ wo 'TimeSeries[{{1, 10}, {5, 50}, {2, 20}}]["Path"]'
{{1, 10}, {2, 20}, {5, 50}}
```
