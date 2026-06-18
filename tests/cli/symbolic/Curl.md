# `Curl`

Curl of a vector field.

```scrut
$ wo 'Curl[{y, -x, 0}, {x, y, z}]'
{0, 0, -2}
```

A third argument selects an orthogonal coordinate system, applying the
appropriate scale factors.

```scrut
$ wo 'Curl[{0, r}, {r, t}, "Polar"]'
2
```

```scrut
$ wo 'Curl[{0, 0, r}, {r, t, z}, "Cylindrical"]'
{0, -1, 0}
```
