# `KeyValueMap`

Applies a two-argument function to each key/value pair.

```scrut
$ wo 'KeyValueMap[List, <|a -> 1, b -> 2|>]'
{{a, 1}, {b, 2}}
```

The operator form `KeyValueMap[f]` applies to an association later.

```scrut
$ wo 'KeyValueMap[List][<|a -> 1, b -> 2|>]'
{{a, 1}, {b, 2}}
```
