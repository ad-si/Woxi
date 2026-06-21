# `Grad`

Gradient of a scalar field.

```scrut
$ wo 'Grad[x^2 y, {x, y}]'
{2*x*y, x^2}
```

For a vector field the result is the Jacobian: row `i` is the gradient of the
`i`-th component.

```scrut
$ wo 'Grad[{x y, x + y}, {x, y}]'
{{y, x}, {1, 1}}
```

A third argument selects an orthogonal coordinate system, applying the
appropriate scale factors.

```scrut
$ wo 'Grad[r Cos[t], {r, t}, "Polar"]'
{Cos[t], -Sin[t]}
```

```scrut
$ wo 'Grad[r^2 z, {r, t, z}, "Cylindrical"]'
{2*r*z, 0, r^2}
```
