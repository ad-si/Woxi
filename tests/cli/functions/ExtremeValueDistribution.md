# `ExtremeValueDistribution`

represents an extreme value distribution.

```scrut
$ wo 'ExtremeValueDistribution[1, 2]'
ExtremeValueDistribution[1, 2]
```

The zero-argument form defaults to location `0` and scale `1`:

```scrut
$ wo 'ExtremeValueDistribution[]'
ExtremeValueDistribution[0, 1]
```

Mean, variance, and median have simple closed forms in the location
parameter `α` and scale parameter `β`:

```scrut
$ wo 'Mean[ExtremeValueDistribution[a, b]]'
a + b*EulerGamma
```

```scrut
$ wo 'Variance[ExtremeValueDistribution[a, b]]'
(b^2*Pi^2)/6
```

```scrut
$ wo 'Median[ExtremeValueDistribution[a, b]]'
a - b*Log[Log[2]]
```
