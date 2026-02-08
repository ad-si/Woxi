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


## `KeyExistsQ`

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; KeyExistsQ[myHash, "Red"]'
True
```


## `KeyDropFrom`

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; KeyDropFrom[myHash, "Green"]'
<|Red -> 1|>
```


## `Keys`

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; Keys[myHash]'
{Green, Red}
```


## `Values`

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; Values[myHash]'
{2, 1}
```


## Nested access

```scrut
$ wo 'assoc = <|"outer" -> <|"inner" -> 8|>|>; assoc["outer", "inner"]'
8
```


## `Map`

```scrut
$ wo 'assoc = <|"a" -> 2, "b" -> 3|>; Map[#^2&, assoc]'
<|a -> 4, b -> 9|>
```


## `AssociationThread`

```scrut
$ wo 'AssociationThread[{a, b, c}, {1, 2, 3}]'
<|a -> 1, b -> 2, c -> 3|>
```


## `Merge`

```scrut
$ wo 'Merge[{<|a -> 1|>, <|a -> 2, b -> 3|>}, Total]'
<|a -> 3, b -> 3|>
```


## `KeyMap`

```scrut
$ wo 'KeyMap[f, <|a -> 1, b -> 2|>]'
<|f[a] -> 1, f[b] -> 2|>
```


## `KeySelect`

```scrut
$ wo 'KeySelect[<|1 -> a, 2 -> b, 3 -> c|>, EvenQ]'
<|2 -> b|>
```


## `KeyTake`

```scrut
$ wo 'KeyTake[<|a -> 1, b -> 2, c -> 3|>, {a, c}]'
<|a -> 1, c -> 3|>
```


## `KeyDrop`

```scrut
$ wo 'KeyDrop[<|a -> 1, b -> 2, c -> 3|>, {a}]'
<|b -> 2, c -> 3|>
```
