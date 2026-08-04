# `FindFit`

Fits a parametric model to numeric data.

```scrut
$ wo 'FindFit[{{1,1},{2,4},{3,9}}, a*x^2, a, x]'
\{a -> 1\.(0+\d*)?\} (regex)
```

A model given as `{model, constraints…}` is fitted subject to those
constraints: the slope here wants to be 2, but is not allowed past 1, so
the intercept takes up the slack.

```scrut
$ wo 'FindFit[{{0, 1.}, {1, 3.}, {2, 5.}, {3, 7.}}, {a x + b, 0 < a < 1}, {a, b}, x]'
\{a -> (1\.(0+\d*)?|0\.99999999\d*), b -> 2\.5(0+\d*)?\} (regex)
```
