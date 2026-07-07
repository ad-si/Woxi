# `LogSeriesDistribution`

Logarithmic series distribution with parameter `θ` (support `k ≥ 1`).

```scrut
$ wo 'LogSeriesDistribution[1/2]'
LogSeriesDistribution[1/2]
```

Probability mass function:

```scrut
$ wo 'PDF[LogSeriesDistribution[1/2], 3]'
1/(24*Log[2])
```

Mean and variance:

```scrut
$ wo 'Mean[LogSeriesDistribution[t]]'
-(t/((1 - t)*Log[1 - t]))
```

```scrut
$ wo 'Variance[LogSeriesDistribution[t]]'
-((t*(t + Log[1 - t]))/((-1 + t)^2*Log[1 - t]^2))
```
