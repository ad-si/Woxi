# `RegionNearest`

Gives the point of a region closest to a given point. For a solid Disk or Ball
an exterior point is projected onto the boundary, while an interior point maps
to itself.

```scrut
$ wo 'RegionNearest[Disk[{0, 0}, 1], {3, 0}]'
{1, 0}
```

```scrut
$ wo 'RegionNearest[Disk[{0, 0}, 2], {3, 4}]'
{6/5, 8/5}
```

A `Rectangle` clamps the point into the box.

```scrut
$ wo 'RegionNearest[Rectangle[{0, 0}, {2, 2}], {3, 3}]'
{2, 2}
```
