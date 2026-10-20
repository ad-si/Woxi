# `CoefficientArrays`

Returns sparse arrays of polynomial coefficients indexed by total degree.

```scrut
$ wo 'Normal[CoefficientArrays[3 x^2 y + 2 x - 7, {x, y}]]'
{-7, {2, 0}, {{0, 0}, {0, 0}}, {{{0, 3}, {0, 0}}, {{0, 0}, {0, 0}}}}
```

Any non-arithmetic expression can serve as a variable:

```scrut
$ wo 'Normal[CoefficientArrays[a f[x]^2 + b f[x] g[x] + c, {f[x], g[x]}]]'
{c, {0, 0}, {{a, b}, {0, 0}}}
```
