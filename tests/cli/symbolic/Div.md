# `Div`

Divergence of a vector field.

```scrut
$ wo 'Div[{x y, y z, z x}, {x, y, z}]'
x + y + z
```

For a tensor field the last index is contracted, so a rank-2 tensor gives a
vector whose `i`-th entry is the divergence of row `i`.

```scrut
$ wo 'Div[{{x, y}, {z, w}}, {x, y}]'
{2, 0}
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
