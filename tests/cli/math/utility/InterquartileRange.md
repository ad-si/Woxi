# `InterquartileRange`

Difference between third and first quartiles.

```scrut
$ wo 'InterquartileRange[{1, 3, 5, 7, 9}]'
5
```

A matrix is reduced columnwise.

```scrut
$ wo 'InterquartileRange[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]'
{2, 20}
```
