# `Conjugate`

Returns the complex conjugate.
For real numbers, returns the number itself.

```scrut
$ wo 'Conjugate[5]'
5
```

The imaginary unit flips sign, and an exact symbolic coefficient stays exact.

```scrut
$ wo 'Conjugate[3 + 4*I]'
3 - 4*I
```

```scrut
$ wo 'Conjugate[2 + I Pi]'
2 - I*Pi
```
