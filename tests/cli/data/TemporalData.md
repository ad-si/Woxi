# `TemporalData`

Builds temporal data from a list of values and a list of time stamps.

A single path of scalar values normalizes to a `TimeSeries`:

```scrut
$ wo 'TemporalData[{2, 1, 6, 5, 7, 4}, {{1, 2, 5, 10, 12, 15}}]'
TimeSeries[{{1, 2}, {2, 1}, {5, 6}, {10, 5}, {12, 7}, {15, 4}}]
```

Descriptive statistics operate on the value path:

```scrut
$ wo 'Mean[TemporalData[{2, 1, 6, 5, 7, 4}, {{1, 2, 5, 10, 12, 15}}]]'
25/6
```

`ListLinePlot` draws the path against its time axis:

```scrut
$ wo 'ListLinePlot[TemporalData[{2, 1, 6, 5, 7, 4}, {{1, 2, 5, 10, 12, 15}}]]'
-Graphics-
```

Several value paths sharing one time axis stay a multi-path `TemporalData`, and
`ListLinePlot` draws one line per path:

```scrut
$ wo 'TemporalData[{{2, 1, 6, 5, 7, 4}, {4, 7, 5, 6, 1, 2}}, {{1, 2, 5, 10, 12, 15}}]'
TemporalData[{{2, 1, 6, 5, 7, 4}, {4, 7, 5, 6, 1, 2}}, {{1, 2, 5, 10, 12, 15}}]
```

```scrut
$ wo 'ListLinePlot[TemporalData[{{2, 1, 6, 5, 7, 4}, {4, 7, 5, 6, 1, 2}}, {{1, 2, 5, 10, 12, 15}}]]'
-Graphics-
```
