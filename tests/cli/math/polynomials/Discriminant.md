# `Discriminant`

Discriminant of a polynomial.

```scrut
$ wo 'Discriminant[x^2 + b x + c, x]'
b^2 - 4*c
```

A degree-0 (constant) polynomial has discriminant `a^(-2)`, with the
zero polynomial special-cased to `0`:

```scrut
$ wo 'Discriminant[3, x]'
1/9
```

```scrut
$ wo 'Discriminant[0, x]'
0
```
