# `Div`

Divergence of a vector field.

```scrut
$ wo 'Div[{x y, y z, z x}, {x, y, z}]'
x + y + z
```

A third argument selects an orthogonal coordinate system, applying the
appropriate scale factors.

```scrut
$ wo 'Div[{r, 0}, {r, t}, "Polar"]'
2
```

```scrut
$ wo 'Div[{r, 0, 0}, {r, t, p}, "Spherical"]'
3
```
