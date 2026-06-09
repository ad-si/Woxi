# `Series`

Computes a power series expansion.

```scrut
$ wo 'Series[Exp[x], {x, 0, 4}]'
SeriesData[x, 0, {1, 1, 1/2, 1/6, 1/24}, 0, 5, 1]
```

Trailing zero coefficients are dropped while the truncation order is kept:

```scrut
$ wo 'Series[1 + x, {x, 0, 3}]'
SeriesData[x, 0, {1, 1}, 0, 4, 1]
```

```scrut
$ wo 'Series[x^2 + x^4, {x, 0, 7}]'
SeriesData[x, 0, {1, 0, 1}, 2, 8, 1]
```

An expression free of the expansion variable is returned unchanged:

```scrut
$ wo 'Series[3, {x, 0, 3}]'
3
```

```scrut
$ wo 'Series[a + b, {x, 0, 3}]'
a + b
```
