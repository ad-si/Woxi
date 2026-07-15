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
