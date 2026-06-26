# `Apart`

Partial-fraction decomposition.

```scrut
$ wo 'Apart[1/((x - 1) (x + 1)), x]'
1/(2*(-1 + x)) - 1/(2*(1 + x))
```

Irreducible quadratic factors are decomposed too:

```scrut
$ wo 'Apart[1/((x^2 + 1) (x^2 + 4))]'
1/(3*(1 + x^2)) - 1/(3*(4 + x^2))
```
