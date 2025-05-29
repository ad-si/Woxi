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
