# Linear Algebra

Matrix and vector operations. Sub-heading levels are
normalised to `##` in this file.

## `Dot`

```scrut
$ wo 'Dot[{1, 2, 3}, {4, 5, 6}]'
32
```

```scrut
$ wo 'Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]'
{{19, 22}, {43, 50}}
```

## `Det`

```scrut
$ wo 'Det[{{1, 2}, {3, 4}}]'
-2
```

## `Inverse`

```scrut
$ wo 'Inverse[{{1, 2}, {3, 4}}]'
{{-2, 1}, {3/2, -1/2}}
```

## `Tr`

```scrut
$ wo 'Tr[{{1, 2}, {3, 4}}]'
5
```

## `IdentityMatrix`

```scrut
$ wo 'IdentityMatrix[3]'
{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
```

## `DiagonalMatrix`

```scrut
$ wo 'DiagonalMatrix[{1, 2, 3}]'
{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}
```

## `Cross`

```scrut
$ wo 'Cross[{1, 2, 3}, {4, 5, 6}]'
{-3, 6, -3}
```


## `Norm`

Vector or matrix norm (Euclidean by default).

```scrut
$ wo 'Norm[{3, 4}]'
5
```

```scrut
$ wo 'Norm[{1, 2, 2}]'
3
```


## `Eigenvalues`

Eigenvalues of a square matrix.

```scrut
$ wo 'Eigenvalues[{{1, 0}, {0, 2}}]'
{2, 1}
```


## `Eigenvectors`

Eigenvectors of a square matrix.

```scrut
$ wo 'Eigenvectors[{{2, 0}, {0, 3}}]'
{{0, 1}, {1, 0}}
```


## `CharacteristicPolynomial`

Characteristic polynomial of a square matrix in a given variable.

```scrut
$ wo 'CharacteristicPolynomial[{{1, 2}, {3, 4}}, x]'
-2 - 5*x + x^2
```


## `MatrixRank`

Rank of a matrix.

```scrut
$ wo 'MatrixRank[{{1, 2}, {2, 4}}]'
1
```


## `MatrixPower`

Integer power of a square matrix.

```scrut
$ wo 'MatrixPower[{{1, 1}, {0, 1}}, 3]'
{{1, 3}, {0, 1}}
```


## `RowReduce`

Gauss–Jordan reduced row-echelon form.

```scrut
$ wo 'RowReduce[{{1, 2}, {3, 4}}]'
{{1, 0}, {0, 1}}
```


## `LinearSolve`

Solves the linear system `m . x == b`.

```scrut
$ wo 'LinearSolve[{{1, 2}, {3, 4}}, {5, 11}]'
{1, 2}
```


## `Adjugate`

Adjugate (classical adjoint) matrix.

```scrut
$ wo 'Adjugate[{{1, 2}, {3, 4}}]'
{{4, -2}, {-3, 1}}
```


