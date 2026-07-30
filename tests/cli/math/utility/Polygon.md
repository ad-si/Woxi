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
