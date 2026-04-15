# Logical Operators and Conditionals

Boolean combinators and the `If` conditional.

## `And`

Logical AND operation.

```scrut
$ wo 'And[True, True]'
True
```

```scrut
$ wo 'And[True, False]'
False
```

```scrut
$ wo 'And[False, True]'
False
```

```scrut
$ wo 'And[False, False]'
False
```

```scrut
$ wo 'And[True, True, True]'
True
```

```scrut
$ wo 'And[True, False, True]'
False
```




## `Or`

Logical OR operation.

```scrut
$ wo 'Or[True, True]'
True
```

```scrut
$ wo 'Or[True, False]'
True
```

```scrut
$ wo 'Or[False, True]'
True
```

```scrut
$ wo 'Or[False, False]'
False
```

```scrut
$ wo 'Or[True, True, True]'
True
```

```scrut
$ wo 'Or[True, False, True]'
True
```




## `Not`

Logical NOT operation.

```scrut
$ wo 'Not[True]'
False
```

```scrut
$ wo 'Not[False]'
True
```

```scrut
$ wo 'Not[True, True]'

Not::argx: Not called with 2 arguments; 1 argument is expected.
.* (regex*)
Not[True, True]
```




## `Xor`

Logical XOR operation.

```scrut
$ wo 'Xor[True, True]'
False
```

```scrut
$ wo 'Xor[True, False]'
True
```

```scrut
$ wo 'Xor[False, True]'
True
```

```scrut
$ wo 'Xor[False, False]'
False
```

```scrut
$ wo 'Xor[True, True, True]'
True
```

```scrut
$ wo 'Xor[True, True, False]'
False
```




## `Nand`

Logical NAND — `Nand[a, b] == Not[And[a, b]]`.

```scrut
$ wo 'Nand[True, True]'
False
```

```scrut
$ wo 'Nand[True, False]'
True
```




## `Nor`

Logical NOR — `Nor[a, b] == Not[Or[a, b]]`.

```scrut
$ wo 'Nor[False, False]'
True
```

```scrut
$ wo 'Nor[True, False]'
False
```




## `Implies`

Logical implication — `False` only when the antecedent is true and
the consequent is false.

```scrut
$ wo 'Implies[True, False]'
False
```

```scrut
$ wo 'Implies[False, True]'
True
```




## `If`

Conditional operation.

```scrut
$ wo 'If[True, 1]'
1
```

```scrut
$ wo 'If[False, 1]'
Null
```

```scrut
$ wo 'If[True, 1, 0]'
1
```

```scrut
$ wo 'If[False, 1, 0]'
0
```

```scrut
$ wo 'If["x", 1, 0, 2]'
2
```

```scrut
$ wo 'If[True, 1, 0, 2, 3]'

If::argb: If called with 5 arguments; between 2 and 4 arguments are expected.
.* (regex*)
If[True, 1, 0, 2, 3]
```




## `Boole`

Maps `True -> 1` and `False -> 0`, threading over lists.

```scrut
$ wo 'Boole[True]'
1
```

```scrut
$ wo 'Boole[False]'
0
```

```scrut
$ wo 'Boole[{True, False, True}]'
{1, 0, 1}
```




## `TrueQ`

Returns `True` only when the argument is literally `True`,
`False` for everything else (including non-Boolean expressions).

```scrut
$ wo 'TrueQ[True]'
True
```

```scrut
$ wo 'TrueQ[1]'
False
```




## `BooleanQ`

Tests whether an expression is `True` or `False`.

```scrut
$ wo 'BooleanQ[True]'
True
```

```scrut
$ wo 'BooleanQ[1]'
False
```




## `SameQ`

Tests structural equality. Unlike `Equal`, `SameQ` returns `False`
when comparing different numeric types (e.g. `1` vs `1.0`).

```scrut
$ wo 'SameQ[1, 1]'
True
```

```scrut
$ wo 'SameQ[1, 1.0]'
False
```




## `UnsameQ`

Negation of `SameQ`.

```scrut
$ wo 'UnsameQ[1, 2]'
True
```

```scrut
$ wo 'UnsameQ[1, 1]'
False
```




## `MatchQ`

Tests whether an expression matches a pattern.

```scrut
$ wo 'MatchQ[5, _Integer]'
True
```

```scrut
$ wo 'MatchQ["hi", _Integer]'
False
```




