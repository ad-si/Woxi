# `LUDecomposition`

LU decomposition with partial pivoting. Wolfram returns the separate lower- and
upper-triangular factors and the row permutation as structured-array objects,
followed by the ∞-norm condition number (`0` for exact matrices).

```scrut
$ wo 'LUDecomposition[{{0, 1}, {1, 0}}]'
{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1, 0}, {0, 1}}]], UpperTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1, 0}, {0, 1}}]], PermutationMatrix[StructuredArray`StructuredData[{2, 2}, {Cycles[{{1, 2}}], Infinity}]], 0}
```

A machine matrix uses magnitude-based pivoting and reports the condition
number:

```scrut
$ wo 'LUDecomposition[{{1., 3.}, {2., 1.}}]'
{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{1., 0.}, {0.5, 1.}}]], UpperTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{2., 1.}, {0., 2.5}}]], PermutationMatrix[StructuredArray`StructuredData[{2, 2}, {Cycles[{{1, 2}}], Infinity}]], 3.2}
```
