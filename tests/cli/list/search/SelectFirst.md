# `SelectFirst`

Returns the first element satisfying a predicate.

```scrut
$ wo 'SelectFirst[{1, 2, 3, 4}, # > 2 &]'
3
```

On an association the predicate tests the values and the first matching value
is returned.

```scrut
$ wo 'SelectFirst[<|a -> 1, b -> 4, c -> 9|>, # > 3 &]'
4
```
