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

`NormFunction` changes the quantity that is minimized. With the 1-norm
the fit follows the median trend instead of being pulled by the outlier.

```scrut
$ wo 'p = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 10}}; Round[{a, b} /. FindFit[p, a x + b, {a, b}, x, NormFunction -> (Norm[#, 1] &)], 0.001]'
{1., 0.}
```
