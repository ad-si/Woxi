# `TrimmedVariance`

Variance after trimming extreme values — the lowest and highest `Floor[f n]`
values are dropped before the variance is taken.

```scrut
$ wo 'TrimmedVariance[{1, 2, 3, 4}, 0.25]'
1/2
```

A pair `{flow, fhigh}` trims the two ends by different fractions:

```scrut
$ wo 'TrimmedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 1000}, {0.1, 0.3}]'
7/2
```

Trimming down to a single value leaves nothing to take a variance of:

```scrut
$ wo 'TrimmedVariance[{1, 2, 3}, 1/3]'

TrimmedVariance::insffnt: There is insufficient data to proceed with the computation.
TrimmedVariance[{1, 2, 3}, 1/3]
```
