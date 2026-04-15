# Basics

## Comments

```scrut
$ wo '(* This comment is ignored *) 5'
5
```

```scrut
$ wo '5 (* This comment is ignored *)'
5
```


## Semicolon

```scrut
$ wo 'x = 2; x'
2
```

```scrut
$ wo 'x = 2; x = x + 5'
7
```

A trailing semicolon evaluates but returns `Null`:

```scrut
$ wo '1 + 2;'
Null
```

```scrut
$ wo '{1,2,3} // Map[Print];'
1
2
3
Null
```


## `Set`

Assign a value to a variable.

```scrut
$ wo 'Set[x, 5]'
5
```

```scrut
$ wo 'Set[x, 5]; x + 3'
8
```


## `Print`

Print values to the console.

```scrut
$ wo 'Print[]'

Null
```

```scrut
$ wo 'Print[5]'
5
Null
```

Multiple arguments are concatenated:

```scrut
$ wo 'Print["a", "b", "c"]'
abc
Null
```

```scrut
$ wo 'Print[1, " + ", 2, " = ", 3]'
1 + 2 = 3
Null
```


## Constants

Woxi ships with the most common mathematical constants
and atomic symbols from the Wolfram Language.
Each constant is a symbolic value that stays exact until you ask for
a numeric approximation with `N`.

### `Pi`

The ratio of a circle's circumference to its diameter.

```scrut
$ wo 'Pi'
Pi
```

```scrut
$ wo 'N[Pi]'
3.141592653589793
```

```scrut
$ wo 'Sin[Pi]'
0
```

### `E`

Euler's number, the base of the natural logarithm.

```scrut
$ wo 'E'
E
```

```scrut
$ wo 'N[E]'
2.718281828459045
```

### `Degree`

The angle unit `Pi/180`, used to convert from degrees to radians.

```scrut
$ wo 'Degree'
Degree
```

```scrut
$ wo 'N[Degree]'
0.017453292519943295
```

### `GoldenRatio`

The golden ratio, `(1 + Sqrt[5])/2`.

```scrut
$ wo 'GoldenRatio'
GoldenRatio
```

```scrut
$ wo 'N[GoldenRatio]'
1.618033988749895
```

### `Catalan`

Catalan's constant, `Sum[(-1)^k/(2k+1)^2, {k, 0, Infinity}]`.

```scrut
$ wo 'Catalan'
Catalan
```

```scrut
$ wo 'N[Catalan]'
0.915965594177219
```

### `EulerGamma`

The Euler–Mascheroni constant.

```scrut
$ wo 'EulerGamma'
EulerGamma
```

```scrut
$ wo 'N[EulerGamma]'
0.5772156649015329
```

### `Glaisher`

The Glaisher–Kinkelin constant.

```scrut
$ wo 'N[Glaisher]'
1.2824271291006226
```

### `Khinchin`

Khinchin's constant, the limit of the geometric mean of the
continued-fraction coefficients of almost every real number.

```scrut
$ wo 'N[Khinchin]'
2.6854520010653062
```

### `Infinity`

Represents a positive unbounded quantity.
Arithmetic with `Infinity` follows the usual conventions.

```scrut
$ wo 'Infinity'
Infinity
```

```scrut
$ wo 'Pi > 3'
True
```

### `ComplexInfinity`

A direction-less unbounded quantity, returned for example by `1/0`.

```scrut
$ wo 'ComplexInfinity'
ComplexInfinity
```

### `Indeterminate`

Represents an expression with no well-defined value, such as `0/0`.

```scrut
$ wo 'Indeterminate'
Indeterminate
```

### `I`

The imaginary unit, satisfying `I^2 == -1`.

```scrut
$ wo 'I'
I
```

```scrut
$ wo 'I^2'
-1
```

```scrut
$ wo 'I^3'
-I
```


## Boolean Atoms

### `True` and `False`

The two Boolean literals.

```scrut
$ wo 'True'
True
```

```scrut
$ wo 'False'
False
```

### `Null`

The empty / absent value.
Expressions terminated by `;` evaluate to `Null`.

```scrut
$ wo 'Null'
Null
```
