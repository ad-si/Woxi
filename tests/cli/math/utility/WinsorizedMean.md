# `WinsorizedMean`

Mean after winsorizing extreme values — the lowest and highest fraction of the
data are replaced by the nearest retained value before averaging.

```scrut
$ wo 'WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]'
11/2
```

A pair `{flow, fhigh}` winsorizes the two ends by different fractions.

```scrut
$ wo 'WinsorizedMean[Range[10], {2/10, 1/10}]'
57/10
```

The fraction defaults to `0.05`, so a one-argument call takes nothing off a
list of ten:

```scrut
$ wo 'WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 1000}]'
209/2
```

An association is averaged over its values:

```scrut
$ wo 'WinsorizedMean[<|"x" -> 1, "y" -> 2, "z" -> 1000|>, 1/3]'
2
```
