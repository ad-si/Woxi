---
icon: lucide/chart-line
---

# Data & Time Series

Woxi can hold and analyze temporal data. A `TimeSeries` pairs values with
time stamps, and descriptive statistics operate on the value path.

```scrut
$ wo 'Mean[TimeSeries[{{1, 10}, {2, 20}, {3, 30}}]]'
20
```

`CompressedData` holds a compressed expression (as produced by `Compress` or
embedded in serialized notebooks) and evaluates to the original on use:

```scrut
$ wo 'Uncompress[Compress[{1, 2, 3}]]'
{1, 2, 3}
```

- [`TimeSeries`](data/TimeSeries.md)
