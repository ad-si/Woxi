# `TriangleCenter`

Compute a named center of a triangle.
The default center is the centroid.

```scrut
$ wo 'TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}]]'
{4/3, 1}
```

```scrut
$ wo 'TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], "Incenter"]'
{1, 1}
```

```scrut
$ wo 'TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], "Circumcenter"]'
{2, 3/2}
```

```scrut
$ wo 'TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], "Orthocenter"]'
{0, 0}
```
