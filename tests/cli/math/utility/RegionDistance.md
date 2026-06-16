# `RegionDistance`

Gives the shortest distance from a point to a region. It is zero inside a solid
region (Disk, Ball) and the boundary distance outside.

```scrut
$ wo 'RegionDistance[Disk[{0, 0}, 1], {3, 0}]'
2
```

```scrut
$ wo 'RegionDistance[Point[{0, 0}], {3, 4}]'
5
```

A `Rectangle` measures the distance to the nearest edge or corner.

```scrut
$ wo 'RegionDistance[Rectangle[{0, 0}, {2, 2}], {3, 3}]'
Sqrt[2]
```
