# `HighlightGraph`

`HighlightGraph[g, spec]` returns the graph `g` with the vertices and edges
named by `spec` drawn in the highlight style — red, by default.

The highlight is a styling only: the graph itself is unchanged.

```scrut
$ wo 'VertexList[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], {2}]]'
{1, 2, 3}
```

The highlighted parts are recorded on the returned graph, so they can be
read back — exactly one vertex is drawn red here.

```scrut
$ wo 'AnnotationValue[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], {2}], GraphHighlight]'
{2}
```

Edges can be highlighted too, and a bare part needs no list.

```scrut
$ wo 'Length[AnnotationValue[HighlightGraph[Graph[{1 <-> 2, 2 <-> 3}], 1 <-> 2], GraphHighlight]]'
1
```

`Style[part, directives]` inside the specification picks the colour for that
part, which is how a graph is coloured by a per-vertex quantity. The parts
themselves are recorded without their styles.

```scrut
$ wo 'g = Graph[{1 <-> 2, 2 <-> 3}]; AnnotationValue[HighlightGraph[g, {Style[1, Green], Style[3, Blue]}], GraphHighlight]'
{1, 3}
```

A subgraph highlights all of its vertices and edges at once — here two
vertices and the edge between them.

```scrut
$ wo 'g = Graph[{1 <-> 2, 2 <-> 3}]; Length[AnnotationValue[HighlightGraph[g, Subgraph[g, {1, 2}]], GraphHighlight]]'
3
```
