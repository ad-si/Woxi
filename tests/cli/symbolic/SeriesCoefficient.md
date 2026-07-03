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

A symbolic index returns the general term as a Piecewise.

```scrut
$ wo 'SeriesCoefficient[Exp[x], {x, 0, n}]'
Piecewise[{{n!^(-1), n >= 0}}, 0]
```

```scrut
$ wo 'SeriesCoefficient[1/(1 - 2 x), {x, 0, n}]'
Piecewise[{{2^n, n >= 0}}, 0]
```
