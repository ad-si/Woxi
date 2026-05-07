# `KeyUnion`

Extends associations to have the union of all keys with Missing for absent keys.

```scrut
$ wo 'KeyUnion[{<|a -> 1|>, <|a -> 2|>}]'
{<|a -> 1|>, <|a -> 2|>}
```
