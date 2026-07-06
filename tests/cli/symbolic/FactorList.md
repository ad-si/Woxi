# `FactorList`

List irreducible factors of a polynomial with exponents.

```scrut
$ wo 'FactorList[6]'
{{6, 1}}
```

A rational number splits into numerator and denominator entries:

```scrut
$ wo 'FactorList[3/4]'
{{3, 1}, {4, -1}}
```

Denominator factors of a rational function carry negative exponents and
come after the numerator factors:

```scrut
$ wo 'FactorList[(x^2 - 1)/(x + 2)]'
{{1, 1}, {-1 + x, 1}, {1 + x, 1}, {2 + x, -1}}
```
