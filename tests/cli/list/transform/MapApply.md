# `MapApply`

Applies the head `f` to the parts of each sublist
(`MapApply[f, list] == Apply[f, #]& /@ list`).

```scrut
$ wo 'MapApply[f, {{1, 2}, {3, 4}}]'
{f[1, 2], f[3, 4]}
```

Any head works, not just `List`.

```scrut
$ wo 'f @@@ g[{a, b}, {c}]'
g[f[a, b], f[c]]
```
