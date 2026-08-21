---
icon: lucide/curly-braces
---

# Associations

```scrut
$ wo '<|"Green" -> 2, "Red" -> 1|>'
<|Green -> 2, Red -> 1|>
```

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>'
<|Green -> 2, Red -> 1|>
```


## Access values in associations

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; myHash[["Green"]]'
2
```

```scrut
$ wo 'myHash = <|1 -> "Red", 2 -> "Green"|>; myHash[[1]]'
Red
```

```scrut
$ wo 'assoc = <|"a" -> True, "b" -> False|>; assoc[["b"]]'
False
```

Calling an association is a lookup, and a lookup composes: each result is
itself callable, and the multi-argument form walks the same chain.

```scrut
$ wo 'a = <|"x" -> <|"y" -> 3|>|>; {a["x"]["y"], a["x", "y"]}'
{3, 3}
```

A key no association holds reports itself:

```scrut
$ wo 'a = <|"x" -> <|"y" -> 3|>|>; a["x", "nope"]'
Missing[KeyAbsent, nope]
```


## Update Values

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; myHash[["Green"]] = 5; myHash'
<|Green -> 5, Red -> 1|>
```


## Add Values

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; myHash[["Puce"]] = 3.5; myHash'
<|Green -> 2, Red -> 1, Puce -> 3.5|>
```

- [`KeyExistsQ`](associations/KeyExistsQ.md)
- [`KeyDropFrom`](associations/KeyDropFrom.md)
- [`Keys`](associations/Keys.md)
- [`Values`](associations/Values.md)
- [`Map`](associations/Map.md)
- [`AssociationThread`](associations/AssociationThread.md)
- [`Merge`](associations/Merge.md)
- [`KeyMap`](associations/KeyMap.md)
- [`KeySelect`](associations/KeySelect.md)
- [`KeyTake`](associations/KeyTake.md)
- [`KeyDrop`](associations/KeyDrop.md)
- [`Association`](associations/Association.md)
- [`Lookup`](associations/Lookup.md)
- [`GroupBy`](associations/GroupBy.md)
- [`KeyValueMap`](associations/KeyValueMap.md)
- [`AssociationMap`](associations/AssociationMap.md)
- [`Length`](associations/Length.md)
- [`Normal`](associations/Normal.md)

## Additional Functions

- [`FilterRules`](associations/FilterRules.md)
- [`KeySort`](associations/KeySort.md)
- [`KeyUnion`](associations/KeyUnion.md)
- [`KeySortBy`](associations/KeySortBy.md)
- [`JoinAcross`](associations/JoinAcross.md)
- [`Dataset`](associations/Dataset.md)
- [`Tabular`](associations/Tabular.md)
- [`ToTabular`](associations/ToTabular.md)
