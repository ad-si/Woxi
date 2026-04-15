# Predicates

Questions about the type, shape, or value of an expression. Predicates in the Wolfram Language conventionally end in `Q`.

## `AllTrue`

Check if all elements in a list satisfy a condition.

```scrut
$ wo 'AllTrue[{2, 4, 6}, EvenQ]'
True
```

```scrut
$ wo 'AllTrue[{2, 3, 4}, EvenQ]'
False
```

```scrut
$ wo 'AllTrue[{1, 3, 6}, 1 <= # <= 6 &]'
True
```


## `ArrayQ`

Tests whether an expression is a regular (rectangular) array
of any rank.

```scrut
$ wo 'ArrayQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'ArrayQ[{{1, 2}, {3, 4}}]'
True
```


## `AssociationQ`

Tests whether an expression is an association.

```scrut
$ wo 'AssociationQ[<|"a" -> 1|>]'
True
```

```scrut
$ wo 'AssociationQ[{1, 2, 3}]'
False
```


## `AtomQ`

Tests whether an expression is atomic — i.e. has no parts.
Strings and numbers are atoms, lists and function calls are not.

```scrut
$ wo 'AtomQ[5]'
True
```

```scrut
$ wo 'AtomQ["hello"]'
True
```

```scrut
$ wo 'AtomQ[{1, 2}]'
False
```


## `Between`

Tests whether a number lies within an inclusive range.

```scrut
$ wo 'Between[5, {1, 10}]'
True
```

```scrut
$ wo 'Between[15, {1, 10}]'
False
```


## `Divisible`

Tests whether the first argument is divisible by the second.

```scrut
$ wo 'Divisible[10, 2]'
True
```

```scrut
$ wo 'Divisible[10, 3]'
False
```


## `DigitQ`

Tests whether a string consists entirely of digit characters.

```scrut
$ wo 'DigitQ["123"]'
True
```

```scrut
$ wo 'DigitQ["12a"]'
False
```


## `DirectoryQ`

Tests whether a given filesystem path is a directory.

```scrut
$ wo 'DirectoryQ["/nonexistent_path_xyz_woxi_test"]'
False
```


## `EvenQ`

Check if a number is even.

```scrut
$ wo 'EvenQ[2]'
True
```

```scrut
$ wo 'EvenQ[3]'
False
```

```scrut
$ wo 'EvenQ[0]'
True
```

```scrut
$ wo 'EvenQ[-4]'
True
```

```scrut
$ wo 'EvenQ[-2]'
True
```

```scrut
$ wo 'EvenQ[100]'
True
```

```scrut
$ wo 'EvenQ[1.5]'
False
```


## `ExactNumberQ`

Tests whether a number is exact (integer, rational, etc. but not a machine float).

```scrut
$ wo 'ExactNumberQ[3]'
True
```

```scrut
$ wo 'ExactNumberQ[1/2]'
True
```

```scrut
$ wo 'ExactNumberQ[3.5]'
False
```


## `FileExistsQ`

Tests whether a given filesystem path exists.

```scrut
$ wo 'FileExistsQ["/nonexistent_path_xyz_woxi_test"]'
False
```


## `FreeQ`

Tests whether an expression does NOT contain a given subexpression.

```scrut
$ wo 'FreeQ[{1, 2, 3}, 4]'
True
```

```scrut
$ wo 'FreeQ[{1, 2, 3}, 2]'
False
```


## `InexactNumberQ`

The negation of `ExactNumberQ`.

```scrut
$ wo 'InexactNumberQ[3.14]'
True
```

```scrut
$ wo 'InexactNumberQ[3]'
False
```


## `IntegerQ`

Check if a value is an integer.

```scrut
$ wo 'IntegerQ[5]'
True
```

```scrut
$ wo 'IntegerQ[0]'
True
```

```scrut
$ wo 'IntegerQ[-7]'
True
```

```scrut
$ wo 'IntegerQ[3.0]'
False
```

```scrut
$ wo 'IntegerQ[3.5]'
False
```

```scrut
$ wo 'IntegerQ[1.2]'
False
```

```scrut
$ wo 'IntegerQ[-0.5]'
False
```

```scrut
$ wo 'IntegerQ[0.0]'
False
```

```scrut
$ wo 'IntegerQ[a]'
False
```


## `LeapYearQ`

Tests whether a given date falls in a leap year.

```scrut
$ wo 'LeapYearQ[{2020, 1, 1}]'
True
```

```scrut
$ wo 'LeapYearQ[{2021, 1, 1}]'
False
```


## `LetterQ`

Tests whether a string consists entirely of letter characters.

```scrut
$ wo 'LetterQ["abc"]'
True
```

```scrut
$ wo 'LetterQ["ab1"]'
False
```


## `ListQ`

Tests whether an expression is a list.

```scrut
$ wo 'ListQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'ListQ[3]'
False
```


## `MachineNumberQ`

Tests whether a value is a machine-precision floating-point number.

```scrut
$ wo 'MachineNumberQ[3.14]'
True
```

```scrut
$ wo 'MachineNumberQ[3]'
False
```


## `MatrixQ`

Tests whether an expression is a rectangular matrix
(a list of equal-length lists).

```scrut
$ wo 'MatrixQ[{{1, 2}, {3, 4}}]'
True
```

```scrut
$ wo 'MatrixQ[{1, 2, 3}]'
False
```


## `MemberQ`

Tests whether a list contains a given element.

```scrut
$ wo 'MemberQ[{1, 2, 3}, 2]'
True
```

```scrut
$ wo 'MemberQ[{1, 2, 3}, 4]'
False
```


## `MissingQ`

Tests whether an expression is a `Missing[...]` wrapper.

```scrut
$ wo 'MissingQ[Missing["NotFound"]]'
True
```

```scrut
$ wo 'MissingQ[5]'
False
```


## `NameQ`

Tests whether a string is the name of a known (built-in or user-defined)
symbol.

```scrut
$ wo 'NameQ["Plus"]'
True
```

```scrut
$ wo 'NameQ["xyz_woxi_nosuch"]'
False
```


## `Negative`

Tests if a number is negative.

```scrut
$ wo 'Negative[-5]'
True
```

```scrut
$ wo 'Negative[3]'
False
```

```scrut
$ wo 'Negative[0]'
False
```


## `NonNegative`

Tests if a number is non-negative (zero or positive).

```scrut
$ wo 'NonNegative[5]'
True
```

```scrut
$ wo 'NonNegative[0]'
True
```

```scrut
$ wo 'NonNegative[-3]'
False
```


## `NonPositive`

Tests if a number is non-positive (zero or negative).

```scrut
$ wo 'NonPositive[-5]'
True
```

```scrut
$ wo 'NonPositive[0]'
True
```

```scrut
$ wo 'NonPositive[3]'
False
```


## `NumberQ`

Tests whether an expression is a (literal) numeric value.

```scrut
$ wo 'NumberQ[3.5]'
True
```

```scrut
$ wo 'NumberQ["abc"]'
False
```


## `NumericQ`

Like `NumberQ` but also recognizes numeric constants such as `Pi`.

```scrut
$ wo 'NumericQ[Pi]'
True
```

```scrut
$ wo 'NumericQ["abc"]'
False
```


## `OddQ`

Check if a number is odd.

```scrut
$ wo 'OddQ[3]'
True
```

```scrut
$ wo 'OddQ[2]'
False
```

```scrut
$ wo 'OddQ[0]'
False
```

```scrut
$ wo 'OddQ[-3]'
True
```

```scrut
$ wo 'OddQ[-4]'
False
```

```scrut
$ wo 'OddQ[101]'
True
```

```scrut
$ wo 'OddQ[2.5]'
False
```


## `Positive`

Tests if a number is positive.

```scrut
$ wo 'Positive[5]'
True
```

```scrut
$ wo 'Positive[-3]'
False
```

```scrut
$ wo 'Positive[0]'
False
```


## `PrimeQ`

Tests whether an integer is prime.

```scrut
$ wo 'PrimeQ[7]'
True
```

```scrut
$ wo 'PrimeQ[8]'
False
```


## `StringQ`

Tests whether an expression is a string.

```scrut
$ wo 'StringQ["hello"]'
True
```

```scrut
$ wo 'StringQ[5]'
False
```


## `VectorQ`

Tests whether an expression is a flat (non-nested) list.

```scrut
$ wo 'VectorQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'VectorQ[{{1, 2}, {3, 4}}]'
False
```




