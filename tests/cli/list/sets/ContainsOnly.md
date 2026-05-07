# `ContainsOnly`

Tests whether every element of a list is contained in another list.

```scrut
$ wo 'ContainsOnly[{1, 2, 1}, {1, 2, 3}]'
True
```

```scrut
$ wo 'ContainsOnly[{1, 2, 4}, {1, 2, 3}]'
False
```
