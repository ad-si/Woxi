# Basic Math Tests

## `Plus`

Add numbers.

```scrut
$ wo 'Plus[1, 2]'
3
```

```scrut
$ wo 'Plus[1, 2, 3]'
6
```

```scrut
$ wo 'Plus[-1, -2]'
-3
```


### `+`

```scrut
$ wo '1+2'
3
```

```scrut
$ wo '8 + (-5)'
3
```


## `Minus`

Make numbers negative.

```scrut
$ wo 'Minus[5]'
-5
```

```scrut
$ wo 'Minus[5, 2]'

Minus::argx: Minus called with 2 arguments; 1 argument is expected.
5 − 2
```


## `Subtract`

Subtract numbers.

```todo
$ wo 'Subtract[5, 2]'
3
```


### `-`

```scrut
$ wo '5-2'
3
```


## `Times`

Multiply numbers.

```scrut
$ wo 'Times[2, 3]'
6
```

```scrut
$ wo 'Times[2, 3, 4]'
24
```

```scrut
$ wo 'Times[-2, 3]'
-6
```


### `*`

```scrut
$ wo '2*2'
4
```

```scrut
$ wo '2 * (-2)'
-4
```


## `Divide`

Divide numbers.

```scrut
$ wo 'Divide[6, 2]'
3
```

```scrut {output_stream: combined}
$ wo 'Divide[6, 2, 3]'

Divide::argrx: Divide called with 3 arguments; 2 arguments are expected.
Divide[6, 2, 3]
```


## `Sign`

Returns the sign of a number.

```scrut
$ wo 'Sign[5]'
1
```

```scrut
$ wo 'Sign[0]'
0
```

```scrut
$ wo 'Sign[-7]'
-1
```


## `Prime`

Returns the nth prime number.

```scrut
$ wo 'Prime[5]'
11
```


## `Abs`

Returns the absolute value of a number.

```scrut
$ wo 'Abs[-5]'
5
```

```scrut
$ wo 'Abs[5]'
5
```

```scrut
$ wo 'Abs[0]'
0
```


## `Floor`

Rounds down to the nearest integer.

```scrut
$ wo 'Floor[3.7]'
3
```

```scrut
$ wo 'Floor[-3.7]'
-4
```

```scrut
$ wo 'Floor[3.2]'
3
```

```scrut
$ wo 'Floor[0]'
0
```

```scrut
$ wo 'Floor[-0]'
0
```

```scrut
$ wo 'Floor[0.5]'
0
```

```scrut
$ wo 'Floor[-0.5]'
-1
```


## `Ceiling`

Rounds up to the nearest integer.

```scrut
$ wo 'Ceiling[3.2]'
4
```

```scrut
$ wo 'Ceiling[-3.2]'
-3
```

```scrut
$ wo 'Ceiling[3.7]'
4
```

```scrut
$ wo 'Ceiling[0]'
0
```

```scrut
$ wo 'Ceiling[-0]'
0
```

```scrut
$ wo 'Ceiling[0.5]'
1
```

```scrut
$ wo 'Ceiling[-0.5]'
0
```


## `Round`

Rounds to the nearest integer.

```scrut
$ wo 'Round[3.5]'
4
```

```scrut
$ wo 'Round[3.4]'
3
```

```scrut
$ wo 'Round[-3.5]'
-4
```

```scrut
$ wo 'Round[-3.4]'
-3
```

```scrut
$ wo 'Round[0.5]'
0
```

```scrut
$ wo 'Round[-0.5]'
0
```

```scrut
$ wo 'Round[0]'
0
```

```scrut
$ wo 'Round[-0]'
0
```


## `Sqrt`

Returns the square root of a number.

```scrut
$ wo 'Sqrt[16]'
4
```

```scrut
$ wo 'Sqrt[0]'
0
```


## `Max`

Returns the maximum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Max[1, 5, 3]'
5
```

```scrut
$ wo 'Max[1, 2]'
2
```

```scrut
$ wo 'Max[-5, -2, -8]'
-2
```

```scrut
$ wo 'Max[3.14, 2.71, 3.5]'
3.5
```

```scrut
$ wo 'Max[1, 2.5, 3]'
3
```

### Single list argument

```scrut
$ wo 'Max[{1, 5, 3}]'
5
```

```scrut
$ wo 'Max[{-10, -5, -20}]'
-5
```

```scrut
$ wo 'Max[{3.14, 2.71, 3.5}]'
3.5
```

### Single value

```scrut
$ wo 'Max[42]'
42
```

```scrut
$ wo 'Max[-7]'
-7
```

### Empty list

```scrut
$ wo 'Max[{}]'
-Infinity
```

### With expressions

```scrut
$ wo 'Max[2 + 3, 4 * 2, 10 - 1]'
9
```

```scrut
$ wo 'Max[{1 + 1, 2 * 2, 3 - 1}]'
4
```


## `Min`

Returns the minimum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Min[1, 5, 3]'
1
```

```scrut
$ wo 'Min[1, 2]'
1
```

```scrut
$ wo 'Min[-5, -2, -8]'
-8
```

```scrut
$ wo 'Min[3.14, 2.71, 3.5]'
2.71
```

```scrut
$ wo 'Min[1, 2.5, 3]'
1
```

### Single list argument

```scrut
$ wo 'Min[{1, 5, 3}]'
1
```

```scrut
$ wo 'Min[{-10, -5, -20}]'
-20
```

```scrut
$ wo 'Min[{3.14, 2.71, 3.5}]'
2.71
```

### Single value

```scrut
$ wo 'Min[42]'
42
```

```scrut
$ wo 'Min[-7]'
-7
```

### Empty list

```scrut
$ wo 'Min[{}]'
Infinity
```

### With expressions

```scrut
$ wo 'Min[2 + 3, 4 * 2, 10 - 1]'
5
```

```scrut
$ wo 'Min[{1 + 1, 2 * 2, 3 - 1}]'
2
```


## `Sin`

Returns the sine of an angle in radians.

```scrut
$ wo 'Sin[Pi/2]'
1
```


## `NumberQ`

Returns `True` if expr is a number, and `False` otherwise.

```scrut
$ wo 'NumberQ[2]'
True
```


## `MemberQ`

Checks if an element is in a list.

```scrut
$ wo 'MemberQ[{1, 2}, 2]'
True
```

```scrut
$ wo 'MemberQ[{1, 2}, 3]'
False
```


## `RandomInteger`

### `RandomInteger[]`

Randomly gives 0 or 1.

```scrut
$ wo 'MemberQ[{0, 1}, RandomInteger[]]'
True
```


### `RandomInteger[{1, 6}]`

Randomly gives a number between 1 and 6.

```scrut
$ wo 'MemberQ[{1, 2, 3, 4, 5, 6}, RandomInteger[{1, 6}]]'
True
```


### `RandomInteger[{1, 6}, 50]`

Randomly gives 50 numbers between 1 and 6.

```scrut
$ wo 'AllTrue[RandomInteger[{1, 6}, 50], 1 <= # <= 6 &]'
True
```
