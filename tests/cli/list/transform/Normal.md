# `Normal`

Converts a sparse array (or other special form) to an explicit nested list.

```scrut
$ wo 'Normal[SparseArray[{1 -> a, 3 -> c}, 4]]'
{a, 0, c, 0}
```
