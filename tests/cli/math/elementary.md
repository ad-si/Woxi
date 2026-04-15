# Elementary Functions

Trigonometric, hyperbolic, exponential, and logarithmic functions.

## `Sin`

Returns the sine of an angle in radians.

```scrut
$ wo 'Sin[Pi/2]'
1
```



## `Log`

Natural logarithm — `Log[b, x]` gives the logarithm of `x` in base `b`.

```scrut
$ wo 'Log[E]'
1
```

```scrut
$ wo 'Log[10, 100]'
2
```


## `Log2`

Logarithm base 2.

```scrut
$ wo 'Log2[8]'
3
```


## `Log10`

Logarithm base 10.

```scrut
$ wo 'Log10[1000]'
3
```


## `Exp`

The exponential function — `Exp[x] == E^x`.

```scrut
$ wo 'Exp[0]'
1
```

```scrut
$ wo 'N[Exp[1]]'
2.718281828459045
```


## `Cos` / `Tan` / `Cot` / `Sec` / `Csc`

The six trigonometric functions.
See [`Sin`](#sin) for related examples.

```scrut
$ wo 'Cos[0]'
1
```

```scrut
$ wo 'Tan[0]'
0
```

```scrut
$ wo 'Cot[Pi/4]'
1
```

```scrut
$ wo 'Sec[0]'
1
```

```scrut
$ wo 'Csc[Pi/2]'
1
```


## `Sinh` / `Cosh` / `Tanh`

Hyperbolic sine, cosine, and tangent.

```scrut
$ wo 'Sinh[0]'
0
```

```scrut
$ wo 'Cosh[0]'
1
```

```scrut
$ wo 'Tanh[0]'
0
```


## `ArcSin` / `ArcCos` / `ArcTan`

Inverse trigonometric functions.
`ArcTan[y, x]` gives the two-argument arctangent.

```scrut
$ wo 'ArcSin[0]'
0
```

```scrut
$ wo 'ArcSin[1]'
Pi/2
```

```scrut
$ wo 'ArcCos[1]'
0
```

```scrut
$ wo 'ArcTan[1]'
Pi/4
```

```scrut
$ wo 'ArcTan[1, 1]'
Pi/4
```


## `ArcSinh` / `ArcCosh` / `ArcTanh`

Inverse hyperbolic functions.

```scrut
$ wo 'ArcSinh[0]'
0
```

```scrut
$ wo 'ArcCosh[1]'
0
```

```scrut
$ wo 'ArcTanh[0]'
0
```


