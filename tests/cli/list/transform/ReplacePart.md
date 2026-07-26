# `ReplacePart`

Replaces element at a specific position.

```scrut
$ wo 'ReplacePart[{a, b, c}, 2 -> x]'
{a, x, c}
```

```scrut
$ wo 'ReplacePart[{1, 2, 3, 4}, 1 -> 0]'
{0, 2, 3, 4}
```

```scrut
$ wo 'ReplacePart[{a, b, c}, -1 -> z]'
{a, b, z}
```

The same replacement can be written with the position last:

```scrut
$ wo 'ReplacePart[{a, b, c}, x, 2]'
{a, x, c}
```

```scrut
$ wo 'ReplacePart[{a, b, c, d}, x, {{1}, {3}}]'
{x, b, x, d}
```
