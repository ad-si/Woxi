# `RootMeanSquare`

Returns the root mean square of a list.

```scrut
$ wo 'RootMeanSquare[{1, 1}]'
1
```

Exact values give an exact root mean square, `Rational`s included.

```scrut
$ wo 'RootMeanSquare[{1/2, 1/3}]'
Sqrt[13/2]/6
```
