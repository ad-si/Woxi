# `RudinShapiro`

Returns `(-1)^k`, where `k` is the number of (possibly overlapping) `11`
pairs in the binary expansion of `n`.

```scrut
$ wo 'RudinShapiro[3]'
-1
```

It threads over a list.

```scrut
$ wo 'RudinShapiro[{1, 2, 3}]'
{1, 1, -1}
```
