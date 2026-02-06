# Syntax


## Postfix application (`//`)

`expr // f` is equivalent to `f[expr]`.
The `//` operator has the lowest precedence, so it applies to the entire
left-hand expression.

```scrut
$ wo '4 // Sqrt'
2
```

```scrut
$ wo '{1, 2, 3} // Length'
3
```

Postfix after an operator chain:

```scrut
$ wo '1 + 2 // ToString'
3
```

```scrut
$ wo 'Sqrt /@ {1, 4, 9} // Length'
3
```

Postfix with a curried function call:

```scrut
$ wo '{1, 4, 9} // Map[Sqrt]'
{1, 2, 3}
```

Chained postfix:

```scrut
$ wo '16 // Sqrt // Sqrt'
2
```


## Prefix application (`@`) with comparison operators

`@` has higher precedence than `==`, so `f@x == y` means `(f@x) == y`.

```scrut
$ wo 'Length@{1,2,3} == 3'
True
```

```scrut
$ wo 'Length@Union@{1,2,1} == Length@Union[{1,2,1} + Range@3] == 3'
False
```


## Alternatives (`|`)

The `|` operator represents alternatives in pattern matching.

```scrut
$ wo 'Cases[{1, "a", 2, "b"}, _Integer]'
{1, 2}
```

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[2 | 4]]'
{1, 3, 5}
```
