# `Dimensions`

Returns the dimensions of a nested list.

```scrut
$ wo 'Dimensions[{{1, 2, 3}, {4, 5, 6}}]'
{2, 3}
```

An association is as long as its number of keys.
Only association values are looked into — a list value is opaque here:

```scrut
$ wo 'Dimensions[<|"a" -> {1, 2}|>]'
{1}
```

```scrut
$ wo 'Dimensions[<|1 -> <|2 -> 3|>|>]'
{1, 1}
```
