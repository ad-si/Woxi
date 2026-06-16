# `StandardDeviation`

Returns the standard deviation of a list.

```scrut
$ wo 'StandardDeviation[{}]'

StandardDeviation::shlen: The argument {} should have at least two elements.
StandardDeviation[{}]
```

For a distribution it returns the symbolic standard deviation. The variance of
a normal distribution is `sigma^2`, so the standard deviation simplifies to
`sigma` (distribution parameters are positive).

```scrut
$ wo 'StandardDeviation[NormalDistribution[mu, sigma]]'
sigma
```

```scrut
$ wo 'StandardDeviation[PoissonDistribution[lambda]]'
Sqrt[lambda]
```
