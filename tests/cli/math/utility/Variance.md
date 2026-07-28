# `Variance`

Returns the variance of a list.

```scrut
$ wo 'Variance[{}]'
Variance[{}]
```

A `Rational` among the values keeps the variance exact; a machine number
anywhere makes it a machine number.

```scrut
$ wo 'Variance[{1/2, 3/2}]'
1/2
```

```scrut
$ wo 'Head[Variance[{1., 2., 3.}]]'
Real
```
