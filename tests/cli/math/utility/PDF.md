# `PDF`

Probability density function.

```scrut
$ wo 'PDF[ChiDistribution[n], x]'
Piecewise[{{(2^(1 - n/2)*x^(-1 + n))/(E^(x^2/2)*Gamma[n/2]), x > 0}}, 0]
```

The value argument threads over a list of points.

```scrut
$ wo 'PDF[PoissonDistribution[3], {1, 2, 3}]'
{3/E^3, 9/(2*E^3), 9/(2*E^3)}
```
