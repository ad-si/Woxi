# `Level`

Returns the sub-expressions at a given level.

```scrut
$ wo 'Level[{{a, b}, {c, d}}, {1}]'
{{a, b}, {c, d}}
```

```scrut
$ wo 'Level[{{a, b}, {c, d}}, {2}]'
{a, b, c, d}
```
