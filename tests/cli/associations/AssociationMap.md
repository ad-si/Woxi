# `AssociationMap`

Builds an association by mapping a function over a list of keys.

```scrut
$ wo 'AssociationMap[f, {a, b, c}]'
<|a -> f[a], b -> f[b], c -> f[c]|>
```

The operator form `AssociationMap[f]` applies to a list of keys later.

```scrut
$ wo 'AssociationMap[f][{a, b, c}]'
<|a -> f[a], b -> f[b], c -> f[c]|>
```
