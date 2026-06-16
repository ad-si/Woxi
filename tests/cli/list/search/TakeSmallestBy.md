# `TakeSmallestBy`

Returns the `n` smallest elements of a list by a key function.

```scrut
$ wo 'TakeSmallestBy[{1, -3, 2, -4, 5}, Abs, 2]'
{1, 2}
```

The operator form `TakeSmallestBy[f, n]` applies to a list later.

```scrut
$ wo 'TakeSmallestBy[Abs, 2][{-3, 1, -8, 4}]'
{1, -3}
```
