# `KeySortBy`

Sorts an association by applying a function to its keys.

```scrut
$ wo 'KeySortBy[<|"b" -> 2, "a" -> 1|>, ToLowerCase]'
<|a -> 1, b -> 2|>
```
