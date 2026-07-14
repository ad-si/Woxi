# `Insphere`

Compute the inscribed sphere of a geometric region.

```scrut
$ wo 'Insphere[Disk[]]'

Insphere::indep: Insphere does not exist for Disk[{0, 0}].
Insphere[Disk[{0, 0}]]
```

The inscribed circle of a triangular 2-simplex (a 3-4-5 right triangle):

```scrut
$ wo 'Insphere[Simplex[{{0, 0}, {4, 0}, {0, 3}}]]'
Sphere[{1, 1}, 1]
```
