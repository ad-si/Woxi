# `DifferenceDelta`

Compute the forward difference `f[x + h] - f[x]` of an expression.

```scrut
$ wo 'DifferenceDelta[x^2, x]'
1 + 2*x
```

A rational summand is combined over a common denominator.

```scrut
$ wo 'DifferenceDelta[1/(2 n + 1), n]'
-2/((1 + 2*n)*(3 + 2*n))
```

With a numeric step the polynomial result is factored.

```scrut
$ wo 'DifferenceDelta[x^2 + x, x]'
2*(1 + x)
```
