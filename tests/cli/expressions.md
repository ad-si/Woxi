# Expressions

In the Wolfram Language everything is an expression
and expressions can be nested.

Woxi supports input in [FullForm] and [InputForm], but not [StandardForm].
Output is always in [FullForm].

[FullForm]: https://reference.wolfram.com/language/ref/InputForm.html
[InputForm]: https://reference.wolfram.com/language/ref/InputForm.html
[StandardForm]: https://reference.wolfram.com/language/ref/StandardForm.html


## `FullForm` - Prints the full form of the expression with no special syntax

```scrut
$ wo 'FullForm[x+y+z]'
Plus[x, y, z]
```

```scrut
$ wo 'FullForm[x y z]'
Times[x, y, z]
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


## `Head` - Returns the head of an expression

```scrut
$ wo 'Head[f[x, y]]'
f
```

```scrut
$ wo 'Head[a + b + c]'
Plus
```

```scrut
$ wo 'Head[{a, b, c}]'
List
```

```scrut
$ wo 'Head[23432]'
Integer
```

```scrut
$ wo 'Head[345.6]'
Real
```


## `Construct` - Constructs an expression from its head and arguments

Apply a function to an argument:

```scrut
$ wo 'Construct[f, x]'
f[x]
```

Apply a function to several arguments:

```scrut
$ wo 'Construct[f, x, y, z]'
f[x, y, z]
```

Build a curried function:

```scrut
$ wo 'Fold[Construct, f, {a, b, c}]'
f[a][b][c]
```


## `TableForm` - Formats data as a table

A one-dimensional list is printed as a column:

```scrut
$ wo 'TableForm[{a, b, c}]'
a
b
c
```

A two-dimensional list is printed as a table:

```scrut
$ wo 'TableForm[{{1, 2, 3}, {4, 5, 6}}]'
1	2	3
4	5	6
```

Columns are right-aligned when they have different widths:

```scrut
$ wo 'TableForm[Table[{i, i^2}, {i, 5}]]'
1	 1
2	 4
3	 9
4	16
5	25
```
