# `MapAll`

Applies a function to every subexpression at every level, like `Map[f, expr, Infinity]`.

```scrut
$ wo 'MapAll[f, {{a, b}, {c, d}}]'
f[{f[{f[a], f[b]}], f[{f[c], f[d]}]}]
```
