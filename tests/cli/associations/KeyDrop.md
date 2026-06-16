# `KeyDrop`

```scrut
$ wo 'KeyDrop[<|a -> 1, b -> 2, c -> 3|>, {a}]'
<|b -> 2, c -> 3|>
```

The operator form `KeyDrop[keys]` applies to an association later.

```scrut
$ wo 'KeyDrop[{b}][<|a -> 1, b -> 2, c -> 3|>]'
<|a -> 1, c -> 3|>
```
