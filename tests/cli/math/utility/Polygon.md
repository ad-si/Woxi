# `Polygon`

Represents a polygon graphics primitive.

```scrut
$ wo 'Head[ToBoxes[Graphics3D[{Polygon[]}]]]'
Graphics3DBox
```

A list of paths draws one polygon each, which is what mapping a face-index
table over a vertex list produces:

```scrut
$ wo 'StringCount[ExportString[Graphics[{Red, Polygon[{{{0, 0}, {1, 0}, {0, 1}}, {{1, 1}, {2, 1}, {1, 2}}}]}], "SVG"], "<polygon"]'
2
```

```scrut
$ wo 'StringCount[ExportString[Graphics[{Red, Polygon[{{0, 0}, {1, 0}, {0, 1}}]}], "SVG"], "<polygon"]'
1
```

`Polygon[outer -> holes]` cuts the hole boundaries out of the face. In SVG
that becomes one path per boundary, filled with the even-odd rule:

```scrut
$ wo 'StringCount[ExportString[Graphics[Polygon[{{0, 0}, {4, 0}, {4, 4}, {0, 4}} -> {{{1, 1}, {3, 1}, {3, 3}, {1, 3}}}]], "SVG"], "evenodd"]'
1
```

A single hole may be given without the enclosing list. In 3D the face is
tessellated around the hole — a square ring becomes eight triangles instead
of the two a solid square gives:

```scrut
$ wo 'StringCount[ExportString[Graphics3D[Polygon[{{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {0, 4, 0}} -> {{1, 1, 0}, {3, 1, 0}, {3, 3, 0}, {1, 3, 0}}], Boxed -> False], "SVG"], "<polygon"]'
8
```
