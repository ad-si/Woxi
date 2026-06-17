# `IndefiniteMatrixQ`

Test whether a matrix is explicitly indefinite, i.e. whether the Hermitian
part of the matrix has both positive and negative eigenvalues.

```scrut
$ wo 'IndefiniteMatrixQ[{{-1, 0}, {0, 1}}]'
True
```

A positive definite matrix is not indefinite:

```scrut
$ wo 'IndefiniteMatrixQ[{{2, 0}, {0, 1}}]'
False
```
