# `Union`

Returns the sorted union of lists (removes duplicates).

```scrut
$ wo 'Union[{1, 2, 3}, {2, 3, 4}]'
{1, 2, 3, 4}
```

```scrut
$ wo 'Union[{a, b}, {b, c}]'
{a, b, c}
```

```scrut
$ wo 'Union[{3, 1, 2}]'
{1, 2, 3}
```

```scrut
$ wo 'Union[{1, 1, 2, 2}]'
{1, 2}
```

```scrut
$ wo 'Union[{1, 2}, {3, 4}, {5, 6}]'
{1, 2, 3, 4, 5, 6}
```
