# `PolynomialExtendedGCD`

Extended GCD of two polynomials: returns `{g, {s, t}}` where `g` is the monic
GCD and `s p + t q == g`.

```scrut
$ wo 'PolynomialExtendedGCD[x^2 - 1, x^2 - 3 x + 2, x]'
{-1 + x, {1/3, -1/3}}
```

```scrut
$ wo 'PolynomialExtendedGCD[x^4 - 1, x^3 - 1, x]'
{-1 + x, {1, -x}}
```

```scrut
$ wo 'PolynomialExtendedGCD[x^2 + 1, x + 1, x]'
{1, {1/2, (1 - x)/2}}
```
