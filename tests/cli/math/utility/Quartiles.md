# `Quartiles`

Returns the quartiles of a list.

```scrut
$ wo 'Quartiles[Range[25]]'
{27/4, 13, 77/4}
```

A matrix is reduced columnwise.

```scrut
$ wo 'Quartiles[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]'
{{3/2, 5/2, 7/2}, {15, 25, 35}}
```
