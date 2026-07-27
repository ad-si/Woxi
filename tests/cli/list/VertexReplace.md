# `VertexReplace`

Renames the vertices of a graph, in its vertex list and in every edge.

```scrut
$ wo 'VertexList[VertexReplace[Graph[{1 <-> 2}], 1 -> 5]]'
{5, 2}
```

```scrut
$ wo 'ToString[EdgeList[VertexReplace[Graph[{1 -> 2}], 2 -> y]], InputForm]'
{DirectedEdge[1, y]}
```

Renaming onto a vertex that is already there merges the two:

```scrut
$ wo 'VertexList[VertexReplace[Graph[{1 <-> 2, 2 <-> 3}], {1 -> 3}]]'
{3, 2}
```
