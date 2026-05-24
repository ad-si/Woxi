# `InverseGammaDistribution`

represents an inverse gamma distribution.

```scrut
$ wo 'InverseGammaDistribution[2, 3]'
InverseGammaDistribution[2, 3]
```

Mean and variance are finite only for shape parameter `α` strictly
greater than `1` and `2` respectively; the Piecewise default is
`Indeterminate`:

```scrut
$ wo 'Mean[InverseGammaDistribution[a, b]]'
Piecewise[{{b/(-1 + a), a > 1}}, Indeterminate]
```

```scrut
$ wo 'Mean[InverseGammaDistribution[3, 2]]'
1
```

The median routes through `InverseGammaRegularized`:

```scrut
$ wo 'Median[InverseGammaDistribution[a, b]]'
b/InverseGammaRegularized[a, 1/2]
```
