# `MultiplySides`

Multiply both sides of an equation by an expression.

```scrut
$ wo 'MultiplySides[x == 2, 3]'
3*x == 6
```

Multiplying by a symbolic factor is guarded, since the step is only
reversible when the factor is non-zero.

```scrut
$ wo 'MultiplySides[a == b, c]'
Piecewise[{{a*c == b*c, c != 0}}, a == b]
```
