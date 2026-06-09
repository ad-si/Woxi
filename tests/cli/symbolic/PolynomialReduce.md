# `PolynomialReduce`

Reduces a polynomial modulo a list of polynomials in a single variable.
`PolynomialReduce[poly, {p1, p2, ...}, x]` returns `{{a1, a2, ...}, b}` with
`a1 p1 + a2 p2 + ... + b == poly` and `b` the minimal remainder.

```scrut
$ wo 'PolynomialReduce[x^2 + 1, {x + 1}, x]'
{{-1 + x}, 2}
```

```scrut
$ wo 'PolynomialReduce[x^3 + 2 x + 1, {x^2 + 1, x + 1}, x]'
{{x, 1}, 0}
```

```scrut
$ wo 'PolynomialReduce[x^3, {x - 1}, x]'
{{1 + x + x^2}, 1}
```

```scrut
$ wo 'PolynomialReduce[x^2, {3 x + 1}, x]'
{{-1/9 + x/3}, 1/9}
```

When no divisor can reduce the polynomial, it is returned as the remainder:

```scrut
$ wo 'PolynomialReduce[x^2 + 1, {x^3 + 1}, x]'
{{0}, 1 + x^2}
```
