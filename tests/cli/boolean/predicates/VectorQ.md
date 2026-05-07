# `VectorQ`

Tests whether an expression is a flat (non-nested) list.

```scrut
$ wo 'VectorQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'VectorQ[{{1, 2}, {3, 4}}]'
False
```
