# `Laplacian`

Laplacian of a scalar field.

```scrut
$ wo 'Laplacian[x^2 + y^2, {x, y}]'
4
```

A third argument selects an orthogonal coordinate system.

```scrut
$ wo 'Laplacian[Log[r], {r, t}, "Polar"]'
0
```

```scrut
$ wo 'Laplacian[r^3, {r, t, p}, "Spherical"]'
12*r
```
