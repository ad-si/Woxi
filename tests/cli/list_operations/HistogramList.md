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
