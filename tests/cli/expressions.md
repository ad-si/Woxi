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


## `TableForm` - Display wrapper (returns unevaluated in text mode)

`TableForm` is a display wrapper that returns unevaluated in text/CLI mode,
matching `wolframscript` behavior:

```scrut
$ wo 'TableForm[{a, b, c}]'
TableForm[{a, b, c}]
```

```scrut
$ wo 'TableForm[{{1, 2, 3}, {4, 5, 6}}]'
TableForm[{{1, 2, 3}, {4, 5, 6}}]
```

Arguments are still evaluated:

```scrut
$ wo 'TableForm[Table[{i, i^2}, {i, 3}]]'
TableForm[{{1, 1}, {2, 4}, {3, 9}}]
```
