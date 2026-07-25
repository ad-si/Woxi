# `PadRight`

Pads a list on the right to a specified length.

```scrut
$ wo 'PadRight[{1, 2, 3}, 5]'
{1, 2, 3, 0, 0}
```

```scrut
$ wo 'PadRight[{a, b}, 4, x]'
{a, b, x, x}
```

```scrut
$ wo 'PadRight[{1, 2, 3, 4, 5}, 3]'
{1, 2, 3}
```

A padding list is repeated cyclically, aligned to the content:

```scrut
$ wo 'PadRight[{1, 2}, 7, {x, y, z}]'
{1, 2, z, x, y, z, x}
```

A padding array shallower than the result lines up with its innermost
levels, so `{x, y}` varies along the columns:

```scrut
$ wo 'PadRight[{{1, 2}, {3, 4}}, {3, 4}, {x, y}, 1]'
{{y, x, y, x}, {y, 1, 2, x}, {y, 3, 4, x}}
```
