# `SparseArray`

Creates a sparse array from position-value rules, a dense list, or with an
explicit default fill value. All forms normalize to the canonical
`SparseArray[Automatic, dims, default, rules]` representation.

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q", {3, 1} -> "Q"}, {3, 3}, "."]]'
{{., Q, .}, {., ., .}, {Q, ., .}}
```

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q"}, {2, 2}, "."]]'
{{., Q}, {., .}}
```

Dimensions are inferred from the maximum position when omitted:

```scrut
$ wo 'SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]'
SparseArray[Automatic, {3, 3}, 0, {1, {{0, 2, 3, 4}, {{1}, {3}, {2}, {3}}}, {1, 4, 2, 3}}]
```

```scrut
$ wo 'Normal[SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]]'
{{1, 0, 4}, {0, 2, 0}, {0, 0, 3}}
```

A dense nested list is converted by recording its non-default entries:

```scrut
$ wo 'SparseArray[{{0, a}, {b, 0}}]'
SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]
```

Arithmetic with scalars or other sparse arrays stays sparse.
Adding a scalar shifts the default fill value instead of densifying:

```scrut
$ wo 'SparseArray[{1 -> 1}, 3] + 1'
SparseArray[Automatic, {3}, 1, {1, {{0, 1}, {{1}}}, {2}}]
```

```scrut
$ wo 'Normal[SparseArray[{1 -> 1}, 3] + 1]'
{2, 1, 1}
```

```scrut
$ wo 'SparseArray[{1 -> 1}, 3] + SparseArray[{2 -> 5}, 3]'
SparseArray[Automatic, {3}, 0, {1, {{0, 2}, {{1}, {2}}}, {1, 5}}]
```

A sparse array answers queries about the grid it stores:

```scrut
$ wo 'sa = SparseArray[{1 -> 5, 3 -> 7}, 4]; {sa["NonzeroValues"], sa["NonzeroPositions"], sa["Density"], sa["Background"]}'
{{5, 7}, {{1}, {3}}, 0.5, 0}
```

The compressed-row structure is reported as it is stored, and the adjacency
lists it stands for are flat for a vector and per row for a matrix:

```scrut
$ wo 'm = SparseArray[{{0, 1}, {2, 0}}]; {m["RowPointers"], m["ColumnIndices"], m["AdjacencyLists"]}'
{{0, 1, 2}, {{2}, {1}}, {{2}, {1}}}
```

It counts as the array it stands for:

```scrut
$ wo '{SparseArrayQ[SparseArray[{1 -> 5}, 3]], ArrayQ[SparseArray[{1 -> 5}, 3]], MatrixQ[SparseArray[{{1, 2}, {3, 4}}]]}'
{True, True, True}
```
