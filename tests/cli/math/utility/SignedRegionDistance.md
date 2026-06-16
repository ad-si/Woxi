# `SignedRegionDistance`

Like `RegionDistance`, but negative for points inside a solid region.

```scrut
$ wo 'SignedRegionDistance[Disk[{0, 0}, 1], {3, 0}]'
2
```

```scrut
$ wo 'SignedRegionDistance[Disk[{0, 0}, 1], {0, 0}]'
-1
```

A `Rectangle` gives the signed axis-aligned-box distance.

```scrut
$ wo 'SignedRegionDistance[Rectangle[{0, 0}, {4, 2}], {1, 1}]'
-1
```
