# `DeleteMissing`

Removes `Missing[...]` elements from a list, or `key -> value` pairs with a
missing value from an association.

```scrut
$ wo 'DeleteMissing[{1, Missing[], 3, Missing["x"], 5}]'
{1, 3, 5}
```

```scrut
$ wo 'DeleteMissing[<|a -> 1, b -> Missing[], c -> 3|>]'
<|a -> 1, c -> 3|>
```
