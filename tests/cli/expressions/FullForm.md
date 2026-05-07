# `FullForm` - Prints the full form of the expression with no special syntax

```scrut
$ wo 'FullForm[x+y+z]'
Plus[x, y, z]
```

```scrut
$ wo 'FullForm[x y z]'
Times[x, y, z]
```

```scrut
$ wo 'FullForm[5*x]'
Times[5, x]
```

```scrut
$ wo 'FullForm[x^n]'
Power[x, n]
```

```scrut
$ wo 'FullForm[{a,b,c}]'
List[a, b, c]
```

```scrut
$ wo 'FullForm[a->b]'
Rule[a, b]
```

<!-- TODO: Why is this not Set[a, b] -->
```scrut
$ wo 'FullForm[a=b]'
b
```

```scrut
$ wo 'FullForm[a b + c]'
Plus[Times[a, b], c]
```

```scrut
$ wo 'FullForm[a (b + c)]'
Times[a, Plus[b, c]]
```
