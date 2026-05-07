# `Values`

```scrut
$ wo 'myHash = <|"Green" -> 2, "Red" -> 1|>; Values[myHash]'
{2, 1}
```


## Nested access

```scrut
$ wo 'assoc = <|"outer" -> <|"inner" -> 8|>|>; assoc["outer", "inner"]'
8
```
