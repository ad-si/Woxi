# `RegionIntersection`

Region common to several regions.

```scrut
$ wo 'RegionIntersection[Rectangle[{0, 0}, {2, 2}], Rectangle[{1, 1}, {3, 3}]]'
Rectangle[{1, 1}, {2, 2}]
```

Regions that do not meet give the empty region of the same embedding
dimension:

```scrut
$ wo 'RegionIntersection[Point[{0, 0}], Point[{1, 1}]]'
EmptyRegion[2]
```

Anything that cannot be combined concretely keeps both regions inside a
`BooleanRegion`:

```scrut
$ wo 'RegionIntersection[Disk[], Disk[{1, 0}]]'
BooleanRegion[#1 && #2 & , {Disk[{0, 0}], Disk[{1, 0}]}]
```

A line segment crossing an infinite plane meets it in a single point:

```scrut
$ wo 'RegionIntersection[InfinitePlane[{0, 0, 1}, {{1, 0, 0}, {0, 1, 0}}], Line[{{0, 0, 0}, {0, 0, 2}}]]'
Point[{{0, 0, 1}}]
```
