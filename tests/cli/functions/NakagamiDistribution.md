# `NakagamiDistribution`

Nakagami distribution with shape parameter `m` and spread `w` (support `x > 0`).

```scrut
$ wo 'NakagamiDistribution[2, 3]'
NakagamiDistribution[2, 3]
```

Probability density function:

```scrut
$ wo 'PDF[NakagamiDistribution[2, 3], 1]'
8/(9*E^(2/3))
```

Variance:

```scrut
$ wo 'Variance[NakagamiDistribution[m, w]]'
w - (w*Pochhammer[m, 1/2]^2)/m
```
