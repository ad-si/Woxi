# `KeySortBy`

Sorts an association by applying a function to its keys.

```scrut
$ wo 'KeySortBy[<|"b" -> 2, "a" -> 1|>, ToLowerCase]'
<|a -> 1, b -> 2|>
```

The sorting function may be a pure function.

```scrut
$ wo 'KeySortBy[<|3 -> a, 1 -> b, 2 -> c|>, -# &]'
<|3 -> a, 2 -> c, 1 -> b|>
```

A list of functions sorts the keys by each criterion in turn.

```scrut
$ wo 'KeySortBy[<|1 -> a, 2 -> b, 3 -> c|>, {Mod[#, 2] &, # &}]'
<|2 -> b, 1 -> a, 3 -> c|>
```
