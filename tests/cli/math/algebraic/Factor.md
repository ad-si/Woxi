# `Factor`

Factors a polynomial.

```scrut
$ wo 'Factor[x^2 - 1]'
(-1 + x)*(1 + x)
```

A non-monic polynomial is factored over the integers via the rational-root
theorem, ordering factors by their leading coefficient.

```scrut
$ wo 'Factor[6 x^2 + 11 x + 3]'
(3 + 2*x)*(1 + 3*x)
```

Multivariate polynomials are factored completely, including repeated factors.

```scrut
$ wo 'Factor[Expand[(a + b + c + d)^3]]'
(a + b + c + d)^3
```

```scrut
$ wo 'Factor[a^3 + b^3 + c^3 - 3 a b c]'
(a + b + c)*(a^2 - a*b + b^2 - a*c - b*c + c^2)
```
