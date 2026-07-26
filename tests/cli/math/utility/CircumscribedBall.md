# `CircumscribedBall`

The ball of minimal radius enclosing a set of points. Exact coordinates give an
exact centre and radius:

```scrut
$ wo 'CircumscribedBall[{{0, 0}, {4, 0}, {0, 3}}]'
Ball[{2, 3/2}, 5/2]
```

Points inside the ball of the others do not enlarge it, so this is the smallest
enclosing ball rather than a circumsphere through every point:

```scrut
$ wo 'CircumscribedBall[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]'
Ball[{1/3, 1/3, 1/3}, Sqrt[2/3]]
```

Vertex-based regions are accepted as well, `Rectangle` and `Cuboid`
contributing all of their corners:

```scrut
$ wo 'CircumscribedBall[Cuboid[{0, 0, 0}, {2, 1, 1}]]'
Ball[{1, 1/2, 1/2}, Sqrt[3/2]]
```

`BoundingRegion[pts, "MinBall"]` is the same ball; `"FastBall"` is the cheaper
circumball of the axis-aligned bounding box:

```scrut
$ wo 'pts = Table[{Mod[k^2, 17], Mod[k^3, 19]}, {k, 1, 12}]; {BoundingRegion[pts, "MinBall"], BoundingRegion[pts, "FastBall"]}'
{Ball[{8, 299/34}, Sqrt[126869]/34], Ball[{17/2, 19/2}, Sqrt[257/2]]}
```
