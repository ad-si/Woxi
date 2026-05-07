# `Complement`

Returns elements in the first list but not in any of the other lists.

```scrut
$ wo 'Complement[{1, 2, 3, 4, 5}, {2, 4}]'
{1, 3, 5}
```

```scrut
$ wo 'Complement[{a, b, c, d}, {b, d}]'
{a, c}
```

```scrut
$ wo 'Complement[{1, 2, 3}, {1, 2, 3}]'
{}
```

```scrut
$ wo 'Complement[{1, 2, 3, 4}, {2}, {4}]'
{1, 3}
```
