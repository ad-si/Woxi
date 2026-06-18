# `NormalMatrixQ`

Test whether a matrix is normal.

```scrut
$ wo 'NormalMatrixQ[{{1, 0}, {0, 2}}]'
True
```

A non-square or non-matrix argument is not normal.

```scrut
$ wo 'NormalMatrixQ[{{1, 2, 3}, {4, 5, 6}}]'
False
```
