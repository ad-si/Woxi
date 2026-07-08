# `Histogram`

Histogram of a list of numeric samples.

```scrut
$ wo 'Head[Histogram[{1, 2, 2, 3, 3, 3}]]'
Graphics
```

A bin specification and a height specification may follow the data.
`{w}` sets the bin width and `"PDF"` normalizes the bars to a probability
density:

```scrut
$ wo 'Head[Histogram[{1, 2, 2, 3, 3, 3}, {1}, "PDF"]]'
Graphics
```

Several datasets (a list of lists) are drawn as overlaid histograms:

```scrut
$ wo 'Head[Histogram[{{1, 2, 2, 3}, {5, 6, 6, 7}}]]'
Graphics
```

Same options as `BarChart`, plus:

- **`Bins`** — `Automatic`, an integer (number of bins), a bin width `{w}`,
  or a list of bin edges `{{e1, e2, ...}}`.
- **Height spec** — `"Count"` (default), `"PDF"`, `"Probability"`,
  `"CumulativeCount"`, or `"CDF"`.
