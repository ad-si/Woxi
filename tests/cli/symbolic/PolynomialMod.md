# `PolynomialMod`

Reduce polynomial coefficients modulo m.

```scrut
$ wo 'PolynomialMod[7, 3]'
1
```

A polynomial modulus divides and keeps the remainder, in the variable it is
written in:

```scrut
$ wo '{PolynomialMod[x^3 + 2 x, x^2 + 1], PolynomialMod[x^2, x + y]}'
{x, y^2}
```

A list of moduli reduces modulo all of them at once — modulo the ideal they
generate, so `7 x^2 + 3` reduces to `10` and then vanishes mod 5:

```scrut
$ wo '{PolynomialMod[7 x^2 + 3, {x^2 - 1, 5}], PolynomialMod[x^4, {x^2 - 2, 3}], PolynomialMod[7, {5, 3}]}'
{0, 1, 0}
```

It threads over a list of polynomials:

```scrut
$ wo 'PolynomialMod[{2 x, 4 x}, 3]'
{2*x, x}
```
