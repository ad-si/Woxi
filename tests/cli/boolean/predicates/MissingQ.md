# `MissingQ`

Tests whether an expression is a `Missing[...]` wrapper.

```scrut
$ wo 'MissingQ[Missing["NotFound"]]'
True
```

```scrut
$ wo 'MissingQ[5]'
False
```
