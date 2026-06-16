# `Counts`

Returns an association mapping each element to its number of occurrences.

```scrut
$ wo 'Counts[{a, b, a, c, b, a}]'
<|a -> 3, b -> 2, c -> 1|>
```

On an association, the values are counted.

```scrut
$ wo 'Counts[<|a -> 1, b -> 1, c -> 2|>]'
<|1 -> 2, 2 -> 1|>
```
