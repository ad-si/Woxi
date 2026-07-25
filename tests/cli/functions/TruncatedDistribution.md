# `TruncatedDistribution`

Restricts a distribution to an interval and renormalizes it,
so the kept part again integrates to 1.

```scrut
$ wo 'PDF[TruncatedDistribution[{1, 3}, UniformDistribution[{0, 4}]], 2]'
1/2
```

Outside the interval the density is zero and the CDF has already reached 1:

```scrut
$ wo 'CDF[TruncatedDistribution[{1, 3}, UniformDistribution[{0, 4}]], 5]'
1
```

Moments follow from the renormalized density:

```scrut
$ wo 'Mean[TruncatedDistribution[{0, 2}, UniformDistribution[{0, 4}]]]'
1
```
