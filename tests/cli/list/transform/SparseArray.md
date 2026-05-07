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
