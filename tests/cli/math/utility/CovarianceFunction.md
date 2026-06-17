# `CovarianceFunction`

Sample autocovariance of a numeric time series at a given lag,
`(1/n) Sum_{t=1}^{n-|h|} (x_t - xbar)(x_{t+|h|} - xbar)`.

```scrut
$ wo 'CovarianceFunction[{2, 3, 4, 3}, 2]'
-1/4
```

Lag zero is the variance (with the `1/n` normalization).

```scrut
$ wo 'CovarianceFunction[{2, 3, 4, 3}, 0]'
1/2
```

The autocovariance is symmetric, so a negative lag equals its magnitude.

```scrut
$ wo 'CovarianceFunction[{1, 2, 3, 4, 5}, -2]'
-1/5
```
