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
