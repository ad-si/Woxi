# `TakeLargestBy`

Returns the `n` largest elements of a list by a key function.

```scrut
$ wo 'TakeLargestBy[{1, -3, 2, -4, 5}, Abs, 2]'
{5, -4}
```

The operator form `TakeLargestBy[f, n]` applies to a list later.

```scrut
$ wo 'TakeLargestBy[Abs, 2][{-3, 1, -8, 4}]'
{-8, 4}
```
