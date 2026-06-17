# `DiagonalMatrixQ`

True if a matrix is diagonal.

```scrut
$ wo 'DiagonalMatrixQ[{{1, 2}, {0, 3}}]'
False
```

`DiagonalMatrixQ[m, k]` tests whether the nonzero entries lie only on the
`k`-th diagonal (positive `k` for super-, negative for sub-diagonals).

```scrut
$ wo 'DiagonalMatrixQ[{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}, 1]'
True
```

Rectangular matrices are accepted.

```scrut
$ wo 'DiagonalMatrixQ[{{1, 0, 0}, {0, 2, 0}}]'
True
```
