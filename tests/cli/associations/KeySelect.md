# `KeySelect`

```scrut
$ wo 'KeySelect[<|1 -> a, 2 -> b, 3 -> c|>, EvenQ]'
<|2 -> b|>
```

The operator form `KeySelect[crit]` applies to an association later.

```scrut
$ wo 'KeySelect[EvenQ][<|1 -> a, 2 -> b, 3 -> c|>]'
<|2 -> b|>
```
