# `CauchyMatrix`

`CauchyMatrix[x, y]` is the matrix with entry (i, j) equal to `1/(x_i + y_j)`.
Wolfram keeps the generating vectors in a structured array:

```scrut
$ wo 'CauchyMatrix[{1, 2, 3}, {1, 5, 6}]'
CauchyMatrix[StructuredArray`StructuredData[{3, 3}, {{1, 2, 3}, {1, 5, 6}}]]
```

`Normal` expands it to the explicit matrix, which may be rectangular:

```scrut
$ wo 'Normal[CauchyMatrix[{1, 2, 3}, {1, 5}]]'
{{1/2, 1/6}, {1/3, 1/7}, {1/4, 1/8}}
```

`CauchyMatrix[x]` is `CauchyMatrix[x, x]`, and `TargetStructure -> "Dense"`
returns the matrix directly:

```scrut
$ wo 'CauchyMatrix[{1, 2, 3}, TargetStructure -> "Dense"]'
{{1/2, 1/3, 1/4}, {1/3, 1/4, 1/5}, {1/4, 1/5, 1/6}}
```

Given an explicit Cauchy matrix, the generating vectors are recovered,
normalized so that `y_1` is `0`:

```scrut
$ wo 'CauchyMatrix[{{1/2, 1/3}, {1/3, 1/4}}]'
CauchyMatrix[StructuredArray`StructuredData[{2, 2}, {{2, 3}, {0, 1}}]]
```
