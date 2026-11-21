# `MinimalPolynomial`

Computes the minimal polynomial of an algebraic number over the rationals.

```scrut
$ wo 'MinimalPolynomial[3, x]'
-3 + x
```

The variable can be any expression; the polynomial is evaluated at it.

```scrut
$ wo 'MinimalPolynomial[Sqrt[2], 1 + y]'
-2 + (1 + y)^2
```
