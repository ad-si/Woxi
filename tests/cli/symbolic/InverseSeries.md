# `InverseSeries`

Computes the reversion of a power series: given the series for `y = f(x)` with
`f(0) = 0` and `f'(0) != 0`, returns the series for the inverse `x = f^(-1)(y)`.

```scrut
$ wo 'InverseSeries[Series[Sin[x], {x, 0, 5}]]'
SeriesData[x, 0, {1, 0, 1/6, 0, 3/40}, 1, 6, 1]
```

```scrut
$ wo 'InverseSeries[Series[Exp[x] - 1, {x, 0, 4}]]'
SeriesData[x, 0, {1, -1/2, 1/3, -1/4}, 1, 5, 1]
```

```scrut
$ wo 'InverseSeries[Series[Tan[x], {x, 0, 6}]]'
SeriesData[x, 0, {1, 0, -1/3, 0, 1/5}, 1, 7, 1]
```

```scrut
$ wo 'InverseSeries[Series[x + x^2, {x, 0, 5}]]'
SeriesData[x, 0, {1, -1, 2, -5, 14}, 1, 6, 1]
```

```scrut
$ wo 'InverseSeries[Series[Sin[x], {x, 0, 5}], y]'
SeriesData[y, 0, {1, 0, 1/6, 0, 3/40}, 1, 6, 1]
```

```scrut
$ wo 'Normal[InverseSeries[Series[Sin[x], {x, 0, 5}]]]'
x + x^3/6 + (3*x^5)/40
```
