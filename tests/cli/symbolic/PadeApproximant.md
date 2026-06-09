# `PadeApproximant`

Computes the `[m/n]` Padé approximant of a function about a point: the rational
function `P(x)/Q(x)` with `deg P <= m`, `deg Q <= n` and `Q(x0) = 1`, whose
power series agrees with the function through order `m + n`.

```scrut
$ wo 'PadeApproximant[Exp[x], {x, 0, {2, 2}}]'
(1 + x/2 + x^2/12)/(1 - x/2 + x^2/12)
```

```scrut
$ wo 'PadeApproximant[Cos[x], {x, 0, {2, 2}}]'
(1 - (5*x^2)/12)/(1 + x^2/12)
```

```scrut
$ wo 'PadeApproximant[Log[1 + x], {x, 0, {2, 2}}]'
(x + x^2/2)/(1 + x + x^2/6)
```

```scrut
$ wo 'PadeApproximant[ArcTan[x], {x, 0, {3, 2}}]'
(x + (4*x^3)/15)/(1 + (3*x^2)/5)
```

With denominator degree `0` the result is just the Taylor polynomial:

```scrut
$ wo 'PadeApproximant[Exp[x], {x, 0, {2, 0}}]'
1 + x + x^2/2
```
