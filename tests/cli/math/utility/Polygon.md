# `Polygon`

Represents a polygon graphics primitive.

```scrut
$ wo 'Head[ToBoxes[Graphics3D[{Polygon[]}]]]'
Graphics3DBox
```

A single path is one face:

```scrut
$ wo 'Area[Polygon[{{0, 0}, {1, 0}, {0, 1}}]]'
1/2
```

A list of paths draws one polygon each — which is what mapping a
face-index table over a vertex list produces — and their areas add up:

```scrut
$ wo 'Area[Polygon[{{{0, 0}, {1, 0}, {0, 1}}, {{1, 1}, {2, 1}, {1, 2}}}]]'
1
```

`Polygon[outer -> holes]` cuts the hole boundaries out of the face, so a
4×4 square with a 2×2 hole measures 12 rather than 16:

```scrut
$ wo 'Area[Polygon[{{0, 0}, {4, 0}, {4, 4}, {0, 4}} -> {{{1, 1}, {3, 1}, {3, 3}, {1, 3}}}]]'
12
```

A single hole may be given without the enclosing list, and the whole
construction works just as well for a face lying on a plane in space:

```scrut
$ wo 'Area[Polygon[{{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {0, 4, 0}} -> {{1, 1, 0}, {3, 1, 0}, {3, 3, 0}, {1, 3, 0}}]]'
12
```

Both forms reach a picture:

```scrut
$ wo 'StringContainsQ[ExportString[Graphics[{Red, Polygon[{{{0, 0}, {1, 0}, {0, 1}}, {{1, 1}, {2, 1}, {1, 2}}}]}], "SVG"], "<svg"]'
True
```

```scrut
$ wo 'StringContainsQ[ExportString[Graphics3D[Polygon[{{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {0, 4, 0}} -> {{1, 1, 0}, {3, 1, 0}, {3, 3, 0}, {1, 3, 0}}], Boxed -> False], "SVG"], "<svg"]'
True
```
