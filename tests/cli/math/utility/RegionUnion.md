# `RegionUnion`

Region covered by any of several regions.

```scrut
$ wo 'RegionUnion[Disk[], Disk[]]'
Disk[{0, 0}]
```

```scrut
$ wo 'RegionUnion[Disk[], Disk[{1, 0}]]'
BooleanRegion[#1 || #2 & , {Disk[{0, 0}], Disk[{1, 0}]}]
```
