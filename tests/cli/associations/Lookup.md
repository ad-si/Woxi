# `Lookup`

Looks up a key in an association, returning a default value if the key is
absent.

```scrut
$ wo 'Lookup[<|a -> 1, b -> 2|>, a]'
1
```

```scrut
$ wo 'Lookup[<|a -> 1, b -> 2|>, c, "missing"]'
missing
```

The operator form `Lookup[key]` applies to an association later.

```scrut
$ wo 'Lookup[a][<|a -> 1, b -> 2|>]'
1
```
