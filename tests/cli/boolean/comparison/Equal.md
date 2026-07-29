# `Equal`

Compare values for equality.

```scrut
$ wo 'Equal[1, 1]'
True
```

```scrut
$ wo 'Equal[1, 2]'
False
```

```scrut
$ wo 'Equal[1, 1, 1]'
True
```


### `==`

Check if values are equal to each other.

```scrut
$ wo '2 == 2'
True
```

```scrut
$ wo 'x = 2; x == 2'
True
```

```scrut
$ wo '2 == 3'
False
```

`Equal` only answers when its operands are comparable. `True`, `False` and
`Null` are each their own kind, so comparing one with a number, a string or a
list has no value and stays as written — it is not `False`:

```scrut
$ wo '1 == True'
1 == True
```

```scrut
$ wo '1 == Null'
1 == Null
```

Comparisons within one kind do resolve:

```scrut
$ wo 'True == False'
False
```

```scrut
$ wo 'Null == Null'
True
```

In a chain, the parts that hold drop away and only the undecidable remainder
comes back:

```scrut
$ wo '2 > 1 == True'
1 == True
```
