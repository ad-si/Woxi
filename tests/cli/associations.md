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


## Set Values

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; myHash[["Green"]] := 5'
Null
```

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; myHash[["Puce"]] := 3.5'
Null
```


## `KeyExistsQ`

```todo
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; KeyExistsQ[myHash, "Red"]'
2
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

```todo
$ wo 'assoc = <|"outer" -> <|"inner" -> 8|>|>; assoc["outer", "inner"]'
8
```


## `Map`

```todo
$ wo 'assoc = <|"a" -> 2, "b" -> 3|>; Map[#^2&, assoc]'
<|a -> 4, b -> 9|>
```
