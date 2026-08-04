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
$ wo 'TimeSeriesWindow[TimeSeries[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}], {3, Infinity}]["Path"]'
{{3, 30}, {4, 40}}
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

`MovingMap` over a series windows by *time* rather than by count: the function
sees the values whose stamps fall in `[t - n, t]`, and the result is stamped
at `t`. Unevenly spaced stamps therefore put different numbers of points in
each window, and one that would reach back past the start of the series is
dropped:

```scrut
$ wo 'MovingMap[Total, TimeSeries[{{1, 1}, {2, 2}, {4, 4}, {7, 7}}], 2]["Path"]'
{{4, 6}, {7, 7}}
```

A plain list still windows by count:

```scrut
$ wo 'MovingMap[Total, {1, 2, 3, 4}, 1]'
{3, 5, 7}
```

## `EventSeries`

An `EventSeries` records events at particular times rather than a sampled
signal. `EventSeriesQ` recognizes one, and `Normal` unwraps it back to
`{time, value}` pairs:

```scrut
$ wo 'EventSeriesQ[EventSeries[{{1, a}, {2, b}}]]'
True
```

```scrut
$ wo 'EventSeriesQ[TimeSeries[{{1, 1}, {2, 2}}]]'
False
```

```scrut
$ wo 'Normal[EventSeries[{{1, a}, {2, b}}]]'
{{1, a}, {2, b}}
```

`EventSeriesLookup[series, t]` gives the events nearest to `t`. A time between
two events picks the closer one, and a time outside the range picks the
nearest end:

```scrut
$ wo 'EventSeriesLookup[EventSeries[{{1, a}, {2, b}, {5, c}}], 3]'
{{2, b}}
```

```scrut
$ wo 'EventSeriesLookup[EventSeries[{{1, a}, {2, b}, {5, c}}], 10]'
{{5, c}}
```

A time exactly between two events is equidistant from both, so both come back:

```scrut
$ wo 'EventSeriesLookup[EventSeries[{{1, a}, {2, b}, {5, c}}], 7/2]'
{{2, b}, {5, c}}
```

`EventSeriesAccumulate` gives the running *count* of events as a `TimeSeries`.
The values themselves play no part — what accumulates is how many events have
occurred:

```scrut
$ wo 'Normal[EventSeriesAccumulate[EventSeries[{{1, 5}, {2, 7}, {5, 9}}]]]'
{{1, 1}, {2, 2}, {5, 3}}
```
