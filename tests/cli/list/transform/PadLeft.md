# `PadLeft`

Pads a list on the left to a specified length.

```scrut
$ wo 'PadLeft[{1, 2, 3}, 5]'
{0, 0, 1, 2, 3}
```

```scrut
$ wo 'PadLeft[{a, b}, 4, x]'
{x, x, a, b}
```

```scrut
$ wo 'PadLeft[{1, 2, 3, 4, 5}, 3]'
{3, 4, 5}
```

A padding list is repeated cyclically, aligned to the content:

```scrut
$ wo 'PadLeft[{1, 2}, 7, {x, y, z}]'
{z, x, y, z, x, 1, 2}
```

For a nested result the padding array tiles it, level by level:

```scrut
$ wo 'PadLeft[{{1, 2}, {3, 4}}, {3, 4}, {{a, b}, {c, d}}]'
{{c, d, c, d}, {a, b, 1, 2}, {c, d, 3, 4}}
```
