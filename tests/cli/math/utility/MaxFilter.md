# `MaxFilter`

Maximum filter over a sliding window.

```scrut
$ wo 'MaxFilter[{1, 5, 2, 8, 3}, 1]'
{5, 5, 8, 8, 8}
```

The window is two-dimensional for a matrix:

```scrut
$ wo 'MaxFilter[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]'
{{5, 6, 6}, {8, 9, 9}, {8, 9, 9}}
```

Unlike the averaging filters, a fractional range is accepted and rounded up:

```scrut
$ wo 'MaxFilter[{1, 2, 3, 4, 100}, 1.5]'
{3, 4, 100, 100, 100}
```
