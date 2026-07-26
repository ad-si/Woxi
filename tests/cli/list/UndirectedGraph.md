# `UndirectedGraph`

`UndirectedGraph[g]` gives the graph with every edge undirected; the
duplicates that leaves are dropped, so a pair of opposite arcs becomes one
edge:

```scrut
$ wo '{EdgeCount[UndirectedGraph[Graph[{1 -> 2, 2 -> 1}]]], EdgeCount[UndirectedGraph[Graph[{1 -> 2, 2 -> 3}]]]}'
{1, 2}
```

The vertices carry over, and the result is undirected:

```scrut
$ wo '{VertexList[UndirectedGraph[Graph[{1 -> 2}]]], UndirectedGraphQ[UndirectedGraph[Graph[{1 -> 2}]]]}'
{{1, 2}, True}
```
