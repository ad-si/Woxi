# `PositiveDefiniteMatrixQ`

Test if a matrix is positive definite.

```scrut
$ wo 'PositiveDefiniteMatrixQ[{{5}}]'
True
```

A non-square or non-matrix argument is not positive definite.

```scrut
$ wo 'PositiveDefiniteMatrixQ[{{1, 2, 3}, {4, 5, 6}}]'
False
```
