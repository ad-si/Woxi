# Expressions

In the Wolfram Language everything is an expression
and expressions can be nested.

Woxi supports input in [FullForm] and [InputForm], but not [StandardForm].
Output is always in [FullForm].

[FullForm]: https://reference.wolfram.com/language/ref/InputForm.html
[InputForm]: https://reference.wolfram.com/language/ref/InputForm.html
[StandardForm]: https://reference.wolfram.com/language/ref/StandardForm.html


## `FullForm` - Prints the full form of the expression with no special syntax

```todo
$ wo 'FullForm[x+y+z]'
Plus[x, y, z]
```

```todo
$ wo 'FullForm[x y z]'
Times[x, y, z]
```

```todo
$ wo 'FullForm[x^n]'
Power[x, n]
```

```todo
$ wo 'FullForm[{a,b,c}]'
List[a, b, c]
```

```todo
$ wo 'FullForm[a->b]'
Rule[a, b]
```

<!-- TODO: Why is this not Set[a, b] -->
```todo
$ wo 'FullForm[a=b]'
b
```

```todo
$ wo 'FullForm[a b + c]'
Plus[Times[a, b], c]
```

```todo
$ wo 'FullForm[a (b + c)]'
Plus[Times[a, b], Times[a, c]]
```


## `Head` - Returns the head of an expression

```todo
$ wo 'Head[f[x, y]]'
f
```

```todo
$ wo 'Head[a + b + c]'
Plus
```

```todo
$ wo 'Head[{a, b, c}]'
List
```

```todo
$ wo 'Head[23432]'
Integer
```

```todo
$ wo 'Head[345.6]'
Real
```


## `Construct` - Constructs an expression from its head and arguments

Apply a function to an argument:

```todo
$ wo 'Construct[f, x]'
f[x]
```

Apply a function to several arguments:

```todo
$ wo 'Construct[f, x, y, z]'
f[x, y, z]
```

Build a curried function:

```todo
$ wo 'Fold[Construct, f, {a, b, c}]'
f[a][b][c]
```
