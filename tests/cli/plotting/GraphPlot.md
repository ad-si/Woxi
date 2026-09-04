# `GraphPlot`

Draws a graph given as a list of edges.

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3, 3 -> 1}]]'
Graphics
```

A square matrix is taken directly as an adjacency matrix — symmetric
matrices give undirected edges, anything else gives directed edges:

```scrut
$ wo 'Head[GraphPlot[{{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}]]'
Graphics
```

```scrut
$ wo 'Head[GraphPlot[{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}]]'
Graphics
```

`Method` names the embedding to lay the vertices out with —
`"CircularEmbedding"` puts all of them on one circle, even when the graph
falls apart into separate pieces:

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 3 -> 4}, Method -> "CircularEmbedding"]]'
Graphics
```

`EdgeShapeFunction -> f` draws each edge with `f[{pt, …}, edge]` instead
of the built-in arrow. The second argument is the edge itself, so a
`DirectedEdge` rather than a plain pair of vertices:

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, EdgeShapeFunction -> ({Blue, Dashed, Line[#1]} &)]]'
Graphics
```

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, EdgeShapeFunction -> (If[MemberQ[{DirectedEdge[1, 2]}, #2], {Blue, Arrow[#1]}, {LightGray, Line[#1]}] &)]]'
Graphics
```

`EdgeShapeFunction -> None` leaves the edges undrawn, a shape name says
how to draw them without a function, and the rule form gives single edges
a shape of their own:

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, EdgeShapeFunction -> None]]'
Graphics
```

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, EdgeShapeFunction -> "Line"]]'
Graphics
```

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, EdgeShapeFunction -> {UndirectedEdge[1, 2] -> ({Red, Line[#1]} &)}]]'
Graphics
```

`VertexRenderingFunction -> f` draws each vertex with `f[{x, y}, name]`
instead of the built-in disk. The first argument is the vertex's
coordinate, the second its name:

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, VertexRenderingFunction -> (Point[#] &)]]'
Graphics
```

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, VertexRenderingFunction -> (Text[#2, #] &)]]'
Graphics
```

`VertexRenderingFunction -> None` leaves the vertices undrawn:

```scrut
$ wo 'Head[GraphPlot[{1 -> 2, 2 -> 3}, VertexRenderingFunction -> None]]'
Graphics
```
