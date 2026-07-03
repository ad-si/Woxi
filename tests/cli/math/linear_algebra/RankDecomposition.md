# `RankDecomposition`

Factors a rank-`r` matrix into `{C, F}`, where `C` holds the pivot columns and
`F` the nonzero reduced-row-echelon rows, so that `C.F` equals the original
matrix.

```scrut
$ wo 'RankDecomposition[{{1, 2}, {2, 4}}]'
{{{1}, {2}}, {{1, 2}}}
```

```scrut
$ wo 'RankDecomposition[{{1, 2, 3}, {4, 5, 6}}]'
{{{1, 2}, {4, 5}}, {{1, 0, -1}, {0, 1, 2}}}
```
