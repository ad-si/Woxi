# Boolean Logic

## `Equal`

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


## `Unequal`

Compare values for inequality.

```scrut
$ wo 'Unequal[1, 1]'
False
```

```scrut
$ wo 'Unequal[1, 2]'
True
```

```scrut
$ wo 'Unequal[1, 1, 1]'
False
```


### `!=`

Check if values are not equal to each other.

```scrut
$ wo '2 != 2'
False
```

```scrut
$ wo 'x = 2; x != 2'
False
```

```scrut
$ wo '2 != 3'
True
```


## `Greater`

Check if values are greater than each other.

```scrut
$ wo 'Greater[2, 1]'
True
```

```scrut
$ wo 'Greater[1, 2]'
False
```

```scrut
$ wo 'Greater[1, 1]'
False
```

```scrut
$ wo 'Greater[1, 2, 3]'
False
```

```scrut
$ wo 'Greater[3, 2, 1]'
True
```


### `>`

Check if values are greater than each other.

```scrut
$ wo '2 > 1'
True
```

```scrut
$ wo '1 > 2'
False
```


## `Less`

Check if values are less than each other.

```scrut
$ wo 'Less[1, 2]'
True
```

```scrut
$ wo 'Less[2, 1]'
False
```

```scrut
$ wo 'Less[1, 1]'
False
```

```scrut
$ wo 'Less[1, 2, 3]'
True
```

```scrut
$ wo 'Less[3, 2, 1]'
False
```


### `<`

Check if values are greater than each other.

```scrut
$ wo '1 < 2'
True
```

```scrut
$ wo '2 < 1'
False
```


## `GreaterEqual`

Check if values are greater than or equal to each other.

```scrut
$ wo 'GreaterEqual[2, 1]'
True
```

```scrut
$ wo 'GreaterEqual[1, 2]'
False
```

```scrut
$ wo 'GreaterEqual[1, 1]'
True
```

```scrut
$ wo 'GreaterEqual[1, 2, 3]'
False
```

```scrut
$ wo 'GreaterEqual[3, 2, 1]'
True
```


### `>=`

```scrut
$ wo '2 >= 1'
True
```

```scrut
$ wo '1 >= 2'
False
```


## `LessEqual`

Check if values are less than or equal to each other.

```scrut
$ wo 'LessEqual[1, 2]'
True
```

```scrut
$ wo 'LessEqual[2, 1]'
False
```

```scrut
$ wo 'LessEqual[1, 1]'
True
```

```scrut
$ wo 'LessEqual[1, 2, 3]'
True
```

```scrut
$ wo 'LessEqual[3, 2, 1]'
False
```


### `<=`

```scrut
$ wo '1 <= 2'
True
```

```scrut
$ wo '2 <= 2'
True
```

```scrut
$ wo '2 <= 1'
False
```


# Multiple comparisons

```scrut
$ wo 'x = 1; 0 <= x <= 2'
True
```

```scrut
$ wo 'x = 1; 0 < x < 2'
True
```

```scrut
$ wo 'x = 1; 0 > x < 2'
False
```

```scrut
$ wo 'x = 1; 0 < x > 2'
False
```


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
If[True, 1, 0, 2, 3]
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
