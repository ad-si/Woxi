# `PositionIndex`

Returns an association mapping each element to the list of its positions.

```scrut
$ wo 'PositionIndex[{a, b, a, c, b}]'
<|a -> {1, 3}, b -> {2, 5}, c -> {4}|>
```

For an association the "positions" are the keys,
so the result maps each distinct value to the keys it is stored under.

```scrut
$ wo 'PositionIndex[<|1 -> a, 2 -> b, 3 -> c, 4 -> d, 5 -> c, 6 -> a|>]'
<|a -> {1, 6}, b -> {2}, c -> {3, 5}, d -> {4}|>
```
