# `FrechetDistribution`

represents a Frechet distribution.

```scrut
$ wo 'FrechetDistribution[2, 3]'
FrechetDistribution[2, 3]
```

Mean and variance exist only for sufficiently large shape parameters
(`α > 1` and `α > 2` respectively); outside those branches the result
is `Infinity`:

```scrut
$ wo 'Mean[FrechetDistribution[a, b]]'
Piecewise[{{b*Gamma[1 - a^(-1)], 1 < a}}, Infinity]
```

```scrut
$ wo 'Mean[FrechetDistribution[2, 3]]'
3*Sqrt[Pi]
```

The median is `β · Log[2]^(-1/α)` for every parameter pair:

```scrut
$ wo 'Median[FrechetDistribution[a, b]]'
b/Log[2]^a^(-1)
```
