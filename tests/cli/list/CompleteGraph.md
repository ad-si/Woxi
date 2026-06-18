# `CompleteGraph`

`CompleteGraph[n]` is the complete graph on `n` vertices — every pair of
vertices is connected, giving `n (n-1)/2` edges.

```scrut
$ wo 'EdgeCount[CompleteGraph[4]]'
6
```

`CompleteGraph[{n1, n2, ...}]` is the complete multipartite graph: vertices are
split into groups of the given sizes and joined only across different groups.

```scrut
$ wo 'EdgeCount[CompleteGraph[{2, 3}]]'
6
```

```scrut
$ wo 'VertexCount[CompleteGraph[{2, 3}]]'
5
```

Such a graph is bipartite.

```scrut
$ wo 'BipartiteGraphQ[CompleteGraph[{2, 3}]]'
True
```
