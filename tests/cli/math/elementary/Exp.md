# `Exp`

The exponential function — `Exp[x] == E^x`.

```scrut
$ wo 'Exp[0]'
1
```

```scrut
$ wo 'N[Exp[1]]'
2.718281828459045
```

Integer and half-integer multiples of `Pi I` evaluate via Euler's formula,
regardless of the factor order.

```scrut
$ wo 'Exp[2 Pi I]'
1
```

```scrut
$ wo 'Exp[3 Pi I/2]'
-I
```
