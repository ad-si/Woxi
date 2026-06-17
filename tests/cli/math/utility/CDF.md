# `CDF`

Cumulative distribution function for a distribution.

```scrut
$ wo 'CDF[ChiDistribution[n], x]'
Piecewise[{{GammaRegularized[n/2, 0, x^2/2], x > 0}}, 0]
```

The value argument threads over a list of points.

```scrut
$ wo 'CDF[PoissonDistribution[3], {1, 2, 3}]'
{4/E^3, 17/(2*E^3), 13/E^3}
```
