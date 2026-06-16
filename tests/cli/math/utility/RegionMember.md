# `RegionMember`

Tests whether a point lies in a region. Disk and Ball are solid (the boundary
is included), while Circle and Sphere are just the boundary curve/surface.

```scrut
$ wo 'RegionMember[Disk[{0, 0}, 1], {0.5, 0.5}]'
True
```

```scrut
$ wo 'RegionMember[Disk[{0, 0}, 1], {2, 0}]'
False
```

```scrut
$ wo 'RegionMember[Rectangle[{0, 0}, {2, 3}], {1, 1}]'
True
```
