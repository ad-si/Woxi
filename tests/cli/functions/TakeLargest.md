# `TakeLargest`

Returns the n largest elements from a list.

```scrut
$ wo 'TakeLargest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]'
{9, 6, 5}
```

```scrut
$ wo 'TakeLargest[{5, 2, 8, 1}, 2]'
{8, 5}
```

The operator form `TakeLargest[n]` applies to a list later.

```scrut
$ wo 'TakeLargest[2][{5, 1, 8, 3}]'
{8, 5}
```
