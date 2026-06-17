# `JacobiP`

Jacobi polynomial.

```scrut
$ wo 'JacobiP[0, 1, 2, 0.5]'
1.
```

It threads over a list of orders.

```scrut
$ wo 'JacobiP[{1, 2}, 0, 0, x]'
{x, (-1 + 3*x^2)/2}
```
