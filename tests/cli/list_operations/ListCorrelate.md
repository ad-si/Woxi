# `ListCorrelate`

Computes discrete cross-correlation of two lists.

```scrut
$ wo 'ListCorrelate[{1, -1}, {1, 2, 4, 8, 16}]'
{-1, -2, -4, -8}
```

A padding list extends the data cyclically:

```scrut
$ wo 'ListCorrelate[{x, y}, {a, b, c, d}, {1, 1}, {p, q}]'
{a*x + b*y, b*x + c*y, c*x + d*y, d*x + p*y}
```

A 5th and 6th argument replace Times and Plus:

```scrut
$ wo 'ListCorrelate[{1, 1}, {1, 2, 3, 4}, {1, -1}, 0, Times, Max]'
{2, 3, 4}
```
