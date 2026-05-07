# `SameQ`

Tests structural equality. Unlike `Equal`, `SameQ` returns `False`
when comparing different numeric types (e.g. `1` vs `1.0`).

```scrut
$ wo 'SameQ[1, 1]'
True
```

```scrut
$ wo 'SameQ[1, 1.0]'
False
```
