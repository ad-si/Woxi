# `GeometricDistribution`

Geometric probability distribution.

```scrut
$ wo 'GeometricDistribution[1/3]'
GeometricDistribution[1/3]
```

The mean is returned in expanded form, while the variance keeps the combined
fraction (matching `wolframscript`).

```scrut
$ wo 'Mean[GeometricDistribution[p]]'
-1 + p^(-1)
```

```scrut
$ wo 'Variance[GeometricDistribution[p]]'
(1 - p)/p^2
```
