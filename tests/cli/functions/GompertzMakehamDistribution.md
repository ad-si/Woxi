# `GompertzMakehamDistribution`

represents a Gompertz-Makeham distribution.

```scrut
$ wo 'GompertzMakehamDistribution[l, x0]'
GompertzMakehamDistribution[l, x0]
```

The two-argument (pure Gompertz) form has closed-form mean and median
in terms of the incomplete gamma function `Gamma[0, ξ]`:

```scrut
$ wo 'Mean[GompertzMakehamDistribution[l, x]]'
(E^x*Gamma[0, x])/l
```

```scrut
$ wo 'Median[GompertzMakehamDistribution[l, x]]'
Log[1 + Log[2]/x]/l
```
