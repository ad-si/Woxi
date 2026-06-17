# `BinormalDistribution`

Bivariate normal distribution. The full form is
`BinormalDistribution[{m1, m2}, {s1, s2}, rho]`; the one-argument form
`BinormalDistribution[rho]` has zero means and unit variances.

```scrut
$ wo 'BinormalDistribution[{m1, m2}, {s1, s2}, r]'
BinormalDistribution[{m1, m2}, {s1, s2}, r]
```

The mean is the location vector:

```scrut
$ wo 'Mean[BinormalDistribution[{m1, m2}, {s1, s2}, r]]'
{m1, m2}
```

The variances are the squared standard deviations:

```scrut
$ wo 'Variance[BinormalDistribution[{m1, m2}, {s1, s2}, r]]'
{s1^2, s2^2}
```

`Covariance` gives the 2×2 covariance matrix:

```scrut
$ wo 'Covariance[BinormalDistribution[{m1, m2}, {s1, s2}, r]]'
{{s1^2, r*s1*s2}, {r*s1*s2, s2^2}}
```
