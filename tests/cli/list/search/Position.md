# `Position`

Finds all positions of a pattern in a list.

```scrut
$ wo 'Position[{a, b, a, c, a}, a]'
{{1}, {3}, {5}}
```

```scrut
$ wo 'Position[{1, 2, 3, 2, 1}, 2]'
{{2}, {4}}
```

```scrut
$ wo 'Position[{1, 2, 3}, 4]'
{}
```

```scrut
$ wo 'Position[{x, y, x, z}, x]'
{{1}, {3}}
```

The four-argument form `Position[expr, pattern, levelspec, n]` returns at
most `n` positions, in scan order:

```scrut
$ wo 'Position[{a, b, a, c, a, b, a}, a, Infinity, 2]'
{{1}, {3}}
```

```scrut
$ wo 'Position[{1, {2, 3}, {4, {5, 6}}}, _Integer, Infinity, 4]'
{{1}, {2, 1}, {2, 2}, {3, 1}}
```
