# `CensoredDistribution`

Clamps a distribution to an interval: values outside are not discarded but
moved onto the nearest endpoint, so the mass piles up there.

The CDF is the base CDF held at 0 below the range and at 1 from its top on:

```scrut
$ wo 'CDF[CensoredDistribution[{1, 3}, UniformDistribution[{0, 4}]], 5]'
1
```

Each moment is the two endpoint atoms plus the base integral between them:

```scrut
$ wo 'Mean[CensoredDistribution[{1, 3}, UniformDistribution[{0, 4}]]]'
2
```

A quantile is the base quantile clamped to the range:

```scrut
$ wo 'Quantile[CensoredDistribution[{1, 3}, UniformDistribution[{0, 4}]], 1/8]'
1
```

There is no ordinary density, since the endpoints carry atoms,
so `PDF` stays unevaluated.
