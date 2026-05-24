# `HalfNormalDistribution`

represents a half-normal distribution with parameter theta.

```scrut
$ wo 'HalfNormalDistribution[1]'
HalfNormalDistribution[1]
```

The mean is `1/theta`:

```scrut
$ wo 'Mean[HalfNormalDistribution[a]]'
a^(-1)
```

The variance is `(Pi - 2) / (2 theta^2)`:

```scrut
$ wo 'Variance[HalfNormalDistribution[a]]'
(-2 + Pi)/(2*a^2)
```

The median is `Sqrt[Pi] InverseErf[1/2] / theta`:

```scrut
$ wo 'Median[HalfNormalDistribution[a]]'
(Sqrt[Pi]*InverseErf[1/2])/a
```
