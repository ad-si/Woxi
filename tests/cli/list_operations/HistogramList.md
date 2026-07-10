# `HistogramList`

Returns bin edges and counts for histogram data.

```scrut
$ wo 'HistogramList[x]'

HistogramList::ldata: x is not a valid dataset or list of datasets.
HistogramList[x]
```

When the data are integer multiples of the bin width, the bins are centered on
the values so none land on a boundary.

```scrut
$ wo 'HistogramList[{1, 2, 2, 3, 3, 3}]'
{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}
```

A bare integer asks for about that many bins: the width `(max-min)/(n-1)` is
floored by the smallest gap between data values and snapped to the nearest
nice number (1, 2 or 5 times a power of ten).

```scrut
$ wo 'HistogramList[{1, 2, 2, 3, 3, 3}, 2]'
{{0, 2, 4}, {1, 5}}
```

```scrut
$ wo 'HistogramList[{1.5, 2.3, 4.7, 8.1, 9.9}, 3]'
{{0, 5, 10}, {3, 2}}
```

```scrut
$ wo 'HistogramList[{0.1, 0.15, 0.2, 0.9}, 4]'
{{0, 1/5, 2/5, 3/5, 4/5, 1}, {2, 1, 0, 0, 1}}
```
