# `SubsetMap`

Map function over subset and recombine.

```scrut
$ wo 'SubsetMap[Reverse, {a, b, c, d, e}, {2, 4}]'
{a, d, c, b, e}
```

The positions may also be given as a span.

```scrut
$ wo 'SubsetMap[Accumulate, {x1, x2, x3, x4, x5, x6}, 2 ;; 5]'
{x1, x2, x2 + x3, x2 + x3 + x4, x2 + x3 + x4 + x5, x6}
```
