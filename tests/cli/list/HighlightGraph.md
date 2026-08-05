# `HighlightGraph`

`HighlightGraph[g, spec]` returns the graph `g` with the vertices and edges
named by `spec` drawn in the highlight style — red, by default.

The highlight is a styling only: the graph itself is unchanged.

```scrut
$ wo 'VertexList[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], {2}]]'
{1, 2, 3}
```

Exactly one vertex is drawn red.

```scrut
$ wo 'StringCount[ExportString[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], {2}], "SVG"], "rgb(255,0,0)"]'
1
```

Edges can be highlighted too, and a bare part needs no list.

```scrut
$ wo 'StringCount[ExportString[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], 1 <-> 2], "SVG"], "rgb(255,0,0)"]'
1
```

`Style[part, directives]` inside the specification picks the colour for that
part, which is how a graph is coloured by a per-vertex quantity.

```scrut
$ wo 'g = Graph[{1 <-> 2, 2 <-> 3}]; StringCount[ExportString[HighlightGraph[g, {Style[1, Green], Style[3, Blue]}], "SVG"], "rgb(0,255,0)"]'
1
```

A subgraph highlights all of its vertices and edges at once.

```scrut
$ wo 'g = Graph[{1 <-> 2, 2 <-> 3}]; StringCount[ExportString[HighlightGraph[g, Subgraph[g, {1, 2}]], "SVG"], "rgb(255,0,0)"]'
3
```
