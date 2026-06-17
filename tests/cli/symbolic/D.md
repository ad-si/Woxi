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

Squaring a derivative distributes the sign correctly, so the Pythagorean
speed of a unit circle collapses cleanly:

```scrut
$ wo 'D[Cos[t], t]^2'
Sin[t]^2
```

```scrut
$ wo 'D[Cos[x], x]^2 + D[Sin[x], x]^2'
Cos[x]^2 + Sin[x]^2
```

Differentiating a power series with respect to its expansion variable applies
the term-by-term power rule:

```scrut
$ wo 'D[Series[Exp[x], {x, 0, 4}], x]'
SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]
```

```scrut
$ wo 'D[SeriesData[x, 0, {1, 1, 1}, 0, 3, 1], x]'
SeriesData[x, 0, {1, 2}, 0, 2, 1]
```
