# `Extract`

Extracts parts at specified positions.

```scrut
$ wo 'Extract[{a, b, c, d}, 2]'
b
```

```scrut
$ wo 'Extract[{a, {b1, b2, b3}, c, d}, {2, 3}]'
b3
```

A part extracted from a held expression evaluates once it leaves the wrapper.

```scrut
$ wo 'Extract[Hold[1 + 2, 3 + 4], {2}]'
7
```
