# `TrimmedMean`

Mean after trimming extreme values. The trim count for each end is
`floor(f * n)`.

```scrut
$ wo 'TrimmedMean[{1, 2, 3, 100}, 0.25]'
5/2
```

A pair `{f1, f2}` trims `f1` of the smallest and `f2` of the largest
elements independently:

```scrut
$ wo 'TrimmedMean[{-10, 1, 1, 1, 1, 20}, {0.2, 0}]'
24/5
```

With no fraction argument, the 5% trimmed mean is used
(`TrimmedMean[list, 0.05]`):

```scrut
$ wo 'TrimmedMean[{-10, 1, 1, 1, 1, 20}]'
7/3
```
