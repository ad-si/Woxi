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

A negative level names the parts of a given depth, counted from the leaves and
measured for each part on its own. `-2` reaches every part whose own depth is
at least two, so the nested `{4, 5}` is mapped but the atoms are not:

```scrut
$ wo 'MapIndexed[f, {{1, 2}, {3, {4, 5}}}, -2]'
{f[{1, 2}, {1}], f[{3, f[{4, 5}, {2, 2}]}, {2}]}
```

`{-1}` on its own reaches only the atoms:

```scrut
$ wo 'MapIndexed[f, {{1, 2}, {3, {4, 5}}}, {-1}]'
{{f[1, {1, 1}], f[2, {1, 2}]}, {f[3, {2, 1}], {f[4, {2, 2, 1}], f[5, {2, 2, 2}]}}}
```

`Infinity` and `All` reach everything:

```scrut
$ wo 'MapIndexed[f, {{1, 2}, {3, {4, 5}}}, Infinity] == MapIndexed[f, {{1, 2}, {3, {4, 5}}}, -1]'
True
```
