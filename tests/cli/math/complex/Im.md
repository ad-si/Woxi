# `Im`

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

An exact symbolic coefficient of `I` is kept exact, not collapsed to a float.

```scrut
$ wo 'Im[I Pi]'
Pi
```

```scrut
$ wo 'Im[5 Pi I/3]'
(5*Pi)/3
```
