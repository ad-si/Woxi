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

The two-argument form additionally requires every element to satisfy a test.

```scrut
$ wo 'VectorQ[{1, 2, x}, NumberQ]'
False
```

```scrut
$ wo 'VectorQ[{1, 2, 3}, Positive]'
True
```
