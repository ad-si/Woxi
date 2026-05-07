# `JoinAcross`

Joins two lists of associations by matching values of a key.

```scrut
$ wo 'JoinAcross[{<|"a" -> 1|>}, {<|"a" -> 1, "b" -> 2|>}, "a"]'
{<|a -> 1, b -> 2|>}
```
