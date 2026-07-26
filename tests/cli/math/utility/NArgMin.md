# `NArgMin`

Numerical arg min.

```scrut
$ wo 'NArgMin[x^2 + 3x + 1, x]'
-1.5
```

With constraints it reports where the optimum sits — one value for a single
variable, a list for several:

```scrut
$ wo '{NArgMin[{-x, x <= 5}, x], NArgMin[{x^2 + y^2, x + y == 2}, {x, y}]}'
{5., {1., 1.}}
```

```scrut
$ wo 'NArgMin[x^2 + y^2, {x, y}]'
{0., 0.}
```
