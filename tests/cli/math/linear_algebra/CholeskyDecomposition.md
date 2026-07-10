# `CholeskyDecomposition`

Returns the upper-triangular Cholesky factor `U` of a Hermitian
positive-definite matrix such that `ConjugateTranspose[U].U` equals the input.

```scrut
$ wo 'CholeskyDecomposition[{{4, 2}, {2, 2}}]'
{{2, 1}, {0, 1}}
```

Exact arithmetic keeps radicals symbolic:

```scrut
$ wo 'CholeskyDecomposition[{{4, 2}, {2, 3}}]'
{{2, 1}, {0, Sqrt[2]}}
```

Machine-precision input yields a numeric factor:

```scrut
$ wo 'CholeskyDecomposition[{{2.0, 1.0}, {1.0, 2.0}}]'
{{1.4142135623730951, 0.7071067811865475}, {0., 1.224744871391589}}
```

Complex Hermitian matrices conjugate the pivot-row factors:

```scrut
$ wo 'CholeskyDecomposition[{{2, I}, {-I, 2}}]'
{{Sqrt[2], I/Sqrt[2]}, {0, Sqrt[3/2]}}
```

`TargetStructure -> "Structured"` wraps the factor as a structured array:

```scrut
$ wo 'CholeskyDecomposition[{{4, 2}, {2, 3}}, TargetStructure -> "Structured"]'
UpperTriangularMatrix[StructuredArray`StructuredData[{2, 2}, {{2, 1}, {0, Sqrt[2]}}]]
```
