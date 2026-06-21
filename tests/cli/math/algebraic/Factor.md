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
