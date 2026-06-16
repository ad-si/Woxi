# `TakeSmallest`

Returns the n smallest elements from a list.

```scrut
$ wo 'TakeSmallest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]'
{1, 1, 2}
```

```scrut
$ wo 'TakeSmallest[{5, 2, 8, 1}, 2]'
{1, 2}
```

The operator form `TakeSmallest[n]` applies to a list later.

```scrut
$ wo 'TakeSmallest[2][{5, 1, 8, 3}]'
{1, 3}
```
