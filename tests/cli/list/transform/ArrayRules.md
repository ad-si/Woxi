# `ArrayRules`

Returns a list of rules describing the non-default entries of a sparse array.

```scrut
$ wo 'ArrayRules[SparseArray[{1 -> a, 3 -> c}, 4]]'
{{1} -> a, {3} -> c, {_} -> 0}
```
