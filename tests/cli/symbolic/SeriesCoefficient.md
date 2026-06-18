# `SeriesCoefficient`

Returns a specific coefficient from a Taylor series.

```scrut
$ wo 'SeriesCoefficient[Sin[x], {x, 0, 3}]'
-1/6
```

The `{x, x0, n}` spec also works on an already-computed series.

```scrut
$ wo 'SeriesCoefficient[Series[Exp[x], {x, 0, 10}], {x, 0, 5}]'
1/120
```
