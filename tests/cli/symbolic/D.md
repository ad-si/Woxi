# `D`

Symbolic differentiation.
`D[expr, x]` differentiates once, `D[expr, {x, n}]` differentiates `n` times,
`D[expr, x, y]` takes mixed partial derivatives.

```scrut
$ wo 'D[Sin[x], x]'
Cos[x]
```

```scrut
$ wo 'D[x^2 + 3 x + 5, x]'
3 + 2*x
```

```scrut
$ wo 'D[x^2 y, {x, 2}]'
2*y
```
