# `FlattenAt`

Flattens a single sublist at a specified position.

```scrut
$ wo 'FlattenAt[{1, {2, 3}, 4, {5, 6}}, 2]'
{1, 2, 3, 4, {5, 6}}
```

Negative indices count from the end:

```scrut
$ wo 'FlattenAt[{a, {b, c}, {d, e}, {f}}, -1]'
{a, {b, c}, {d, e}, f}
```

Flatten at several positions:

```scrut
$ wo 'FlattenAt[{a, {b, c}, {d, e}, {f}}, {{2}, {4}}]'
{a, b, c, {d, e}, f}
```

Flatten at a nested position:

```scrut
$ wo 'FlattenAt[{a, {{b, c}, {d, e}}, {f}}, {2, 1}]'
{a, {b, c, {d, e}}, {f}}
```

Operator form:

```scrut
$ wo 'FlattenAt[2][{a, {b, c}, {d, e}, {f}}]'
{a, b, c, {d, e}, {f}}
```
