# `Histogram`

Histogram of a list of numeric samples.

```scrut
$ wo 'Head[Histogram[{1, 2, 2, 3, 3, 3}]]'
Graphics
```

Same options as `BarChart`, plus:

- **`Bins`** — `Automatic`, an integer (number of bins),
  or a list of bin edges.
