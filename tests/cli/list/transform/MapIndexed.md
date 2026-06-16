# `MapIndexed`

Applies a function to each element and its index.

```scrut
$ wo 'MapIndexed[f, {a, b, c}]'
{f[a, {1}], f[b, {2}], f[c, {3}]}
```

On an association the index is `{Key[key]}`.

```scrut
$ wo 'MapIndexed[f, <|"x" -> 10, "y" -> 20|>]'
<|x -> f[10, {Key[x]}], y -> f[20, {Key[y]}]|>
```
