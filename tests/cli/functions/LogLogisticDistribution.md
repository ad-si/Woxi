# `LogLogisticDistribution`

Log-logistic distribution with shape parameter `g` and scale `s` (support `x > 0`).

```scrut
$ wo 'LogLogisticDistribution[2, 3]'
LogLogisticDistribution[2, 3]
```

Probability density function:

```scrut
$ wo 'PDF[LogLogisticDistribution[2, 3], 1]'
9/50
```

Cumulative distribution function:

```scrut
$ wo 'CDF[LogLogisticDistribution[2, 3], 1]'
1/10
```

Mean:

```scrut
$ wo 'Mean[LogLogisticDistribution[g, s]]'
Piecewise[{{(Pi*s*Csc[Pi/g])/g, g > 1}}, Indeterminate]
```
