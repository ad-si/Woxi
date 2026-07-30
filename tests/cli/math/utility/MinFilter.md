# `MinFilter`

Minimum filter over a sliding window.

```scrut
$ wo 'MinFilter[{1, 5, 2, 8, 3}, 1]'
{1, 1, 2, 2, 3}
```

The window is two-dimensional for a matrix:

```scrut
$ wo 'MinFilter[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]'
{{1, 1, 2}, {1, 1, 2}, {4, 4, 5}}
```

A fractional range rounds up and the sign is ignored, so these are all the
radius-2 filter:

```scrut
$ wo 'MinFilter[{1, 2, 3, 4, 100}, 1.5] == MinFilter[{1, 2, 3, 4, 100}, -2]'
True
```

A range that names no neighborhood is reported:

```scrut
$ wo 'MinFilter[{1, 2, 3, 4, 100}, x]'

MinFilter::bdrad: x is not a valid neighborhood range specification.
MinFilter[{1, 2, 3, 4, 100}, x]
```
