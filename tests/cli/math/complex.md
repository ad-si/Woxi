# Complex Numbers and Number Predicates

Accessors for complex numbers, plus `NumberQ` / `NumericQ` and related
helpers for checking and converting numeric values.

## `NumericQ`

Tests if an expression has a numeric value.

```scrut
$ wo 'NumericQ[42]'
True
```

```scrut
$ wo 'NumericQ["hello"]'
False
```

```scrut
$ wo 'NumericQ[3.14]'
True
```


## `NumberQ`

Returns `True` if expr is a number, and `False` otherwise.

```scrut
$ wo 'NumberQ[2]'
True
```


## `Re`

Returns the real part of a number.
For real numbers, returns the number itself.

```scrut
$ wo 'Re[5]'
5
```

```scrut
$ wo 'Re[3.14]'
3.14
```

```scrut
$ wo 'Re[3 + 4*I]'
3
```


## `Im`

Returns the imaginary part of a number.
For real numbers, returns 0.

```scrut
$ wo 'Im[5]'
0
```

```scrut
$ wo 'Im[3.14]'
0
```

```scrut
$ wo 'Im[3 + 4*I]'
4
```


## `Conjugate`

Returns the complex conjugate.
For real numbers, returns the number itself.

```scrut
$ wo 'Conjugate[5]'
5
```


## `Arg`

Returns the argument (phase angle) of a complex number in radians.

```scrut
$ wo 'Arg[1]'
0
```

```scrut
$ wo 'Arg[-1]'
Pi
```

```scrut
$ wo 'Arg[0]'
0
```


## `Rationalize`

Converts a decimal number to a rational approximation.

```scrut
$ wo 'Rationalize[0.5]'
1/2
```

```scrut
$ wo 'Rationalize[0.333333]'
0.333333
```

```scrut
$ wo 'Rationalize[0.25]'
1/4
```

```scrut
$ wo 'Rationalize[3]'
3
```

With a tolerance argument, finds a rational within that tolerance:

```scrut
$ wo 'Rationalize[0.333333, 0.0001]'
1/3
```

```scrut
$ wo 'Rationalize[0.333333, 0.00001]'
1/3
```

Numbers with up to 5 decimal places are rationalized:

```scrut
$ wo 'Rationalize[0.33333]'
33333/100000
```


