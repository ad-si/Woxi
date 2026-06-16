# `KeyMap`

```scrut
$ wo 'KeyMap[f, <|a -> 1, b -> 2|>]'
<|f[a] -> 1, f[b] -> 2|>
```

The operator form `KeyMap[f]` applies to an association later.

```scrut
$ wo 'KeyMap[f][<|a -> 1, b -> 2|>]'
<|f[a] -> 1, f[b] -> 2|>
```
