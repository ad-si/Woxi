# `MeshRegion`

A region given by coordinates and the cells between them, as
`ConvexHullMesh`, `DelaunayMesh` and `VoronoiMesh` build it. The ordinary
region functions read it, and exact coordinates keep exact answers.

```scrut
$ wo 'RegionMeasure[ConvexHullMesh[{{0, 0}, {2, 0}, {0, 2}}]]'
2
```

```scrut
$ wo 'Perimeter[ConvexHullMesh[{{0, 0}, {2, 0}, {0, 2}}]]'
4 + 2*Sqrt[2]
```

```scrut
$ wo 'RegionCentroid[ConvexHullMesh[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]]'
{1, 1}
```

```scrut
$ wo 'RegionMember[ConvexHullMesh[{{0, 0}, {2, 0}, {0, 2}}], {0.5, 0.5}]'
True
```

`MeshCoordinates`, `MeshCells`, `MeshPrimitives` and `MeshCellCount` read the
object itself. Cells are named by the indices of the coordinates they join,
and primitives write those coordinates out:

```scrut
$ wo 'MeshCells[ConvexHullMesh[{{0, 0}, {2, 0}, {0, 2}}], 1]'
{Line[{1, 2}], Line[{2, 3}], Line[{3, 1}]}
```

```scrut
$ wo 'MeshPrimitives[ConvexHullMesh[{{0, 0}, {2, 0}, {0, 2}}], 1]'
{Line[{{0, 0}, {2, 0}}], Line[{{2, 0}, {0, 2}}], Line[{{0, 2}, {0, 0}}]}
```

```scrut
$ wo 'MeshCellCount[DelaunayMesh[{{0, 0}, {2, 0}, {0, 2}, {2, 2}}]]'
{4, 5, 2}
```

`DelaunayMesh` triangulates a point set:

```scrut
$ wo 'RegionMeasure[DelaunayMesh[{{0, 0}, {2, 0}, {0, 2}, {2, 2}}]]'
4
```

```scrut
$ wo 'Sort[Sort /@ MeshCells[DelaunayMesh[{{0, 0}, {2, 0}, {0, 2}, {2, 2}}], 2][[All, 1]]]'
{{1, 2, 3}, {2, 3, 4}}
```
