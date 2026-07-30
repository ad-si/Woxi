# `WinsorizedVariance`

Variance after winsorizing extreme values — the lowest and highest `Floor[f n]`
values are replaced by the nearest retained value, so the count is unchanged.

```scrut
$ wo 'WinsorizedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 1000}, {0.1, 0.3}]'
40/9
```

Because nothing is dropped, winsorizing always leaves enough data:

```scrut
$ wo 'WinsorizedVariance[{1, 2, 3}, 1/3]'
0
```

A matrix is reduced column by column:

```scrut
$ wo 'WinsorizedVariance[{{1, 2}, {3, 4}, {5, 100}}, 1/3]'
{0, 0}
```
