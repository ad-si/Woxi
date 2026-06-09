# `ComposeSeries`

Composes power series: `ComposeSeries[s1, s2]` gives the series for the
substitution `s1(s2(x))`. The inner series must have a zero constant term.

```scrut
$ wo 'ComposeSeries[Series[Exp[x], {x, 0, 3}], Series[Sin[x], {x, 0, 3}]]'
SeriesData[x, 0, {1, 1, 1/2}, 0, 4, 1]
```

```scrut
$ wo 'ComposeSeries[Series[Exp[y], {y, 0, 4}], Series[x + x^2, {x, 0, 4}]]'
SeriesData[x, 0, {1, 1, 3/2, 7/6, 25/24}, 0, 5, 1]
```

```scrut
$ wo 'ComposeSeries[Series[1/(1-y), {y, 0, 2}], Series[x^2, {x, 0, 5}]]'
SeriesData[x, 0, {1, 0, 1}, 0, 4, 1]
```

```scrut
$ wo 'ComposeSeries[Series[Log[1+y], {y, 0, 4}], Series[x + x^3, {x, 0, 4}]]'
SeriesData[x, 0, {1, -1/2, 4/3, -5/4}, 1, 5, 1]
```

Composition is associative across several series:

```scrut
$ wo 'ComposeSeries[Series[Exp[x], {x, 0, 3}], Series[Sin[y], {y, 0, 3}], Series[z^1, {z, 0, 3}]]'
SeriesData[z, 0, {1, 1, 1/2}, 0, 4, 1]
```
