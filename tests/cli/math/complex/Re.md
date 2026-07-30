# `Re`

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

A complex exponential resolves to its trigonometric form:

```scrut
$ wo 'Re[2 E^(Pi I/25)]'
2*Cos[Pi/25]
```
