# `KeySort`

Sorts an association by its keys.

```scrut
$ wo 'KeySort[<|a -> 1, b -> 2|>]'
<|a -> 1, b -> 2|>
```

A second argument orders the keys with a comparison function.

```scrut
$ wo 'KeySort[<|3 -> a, 1 -> b, 2 -> c|>, Greater]'
<|3 -> a, 2 -> c, 1 -> b|>
```
