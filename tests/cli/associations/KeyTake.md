# `KeyTake`

```scrut
$ wo 'KeyTake[<|a -> 1, b -> 2, c -> 3|>, {a, c}]'
<|a -> 1, c -> 3|>
```

The operator form `KeyTake[keys]` applies to an association later.

```scrut
$ wo 'KeyTake[{a, c}][<|a -> 1, b -> 2, c -> 3|>]'
<|a -> 1, c -> 3|>
```
