# `LDLDecomposition`

Returns the L·D·Lᴴ factorization of a Hermitian (or real symmetric) matrix:
`L` unit lower triangular and `D` diagonal, as structured arrays.

```scrut
$ wo 'LDLDecomposition[{{4, 2}, {2, 3}}]'
{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1, 0}, {1/2, 1}}]], DiagonalMatrix[StructuredArray`StructuredData[{2, 2}, {{4, 2}, 0}]]}
```

Unlike Cholesky, indefinite matrices work (D picks up negative entries):

```scrut
$ wo 'LDLDecomposition[{{1, 2}, {2, 1}}]'
{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1, 0}, {2, 1}}]], DiagonalMatrix[StructuredArray`StructuredData[{2, 2}, {{1, -3}, 0}]]}
```

Complex Hermitian matrices stay exact:

```scrut
$ wo 'LDLDecomposition[{{2, I}, {-I, 2}}]'
{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1, 0}, {-1/2*I, 1}}]], DiagonalMatrix[StructuredArray`StructuredData[{2, 2}, {{2, 3/2}, 0}]]}
```

`TargetStructure -> "Dense"` returns plain matrices:

```scrut
$ wo 'LDLDecomposition[{{4, 2}, {2, 3}}, TargetStructure -> "Dense"]'
{{{1, 0}, {1/2, 1}}, {{4, 0}, {0, 2}}}
```
