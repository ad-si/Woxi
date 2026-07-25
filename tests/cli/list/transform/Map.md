# `Map`

Applies a function to each element of a list.

```scrut
$ wo 'Map[Sign, {-6, 0, 2, 5}]'
{-1, 0, 1, 1}
```

The short form `/@` threads over any head, not just lists.

```scrut
$ wo 'f /@ g[a, b]'
g[f[a], f[b]]
```

An atom has no parts, so mapping leaves it unchanged.

```scrut
$ wo 'f /@ 5'
5
```
