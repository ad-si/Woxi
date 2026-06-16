# `CurryApplied`

`CurryApplied[f, n]` is an operator that collects `n` arguments supplied one
at a time and applies `f` to them in order.

```scrut
$ wo 'CurryApplied[f, 2][a][b]'
f[a, b]
```

```scrut
$ wo 'CurryApplied[Plus, 3][1][2][3]'
6
```

A list second argument gives an explicit argument permutation.

```scrut
$ wo 'CurryApplied[f, {2, 1}][a][b]'
f[b, a]
```
