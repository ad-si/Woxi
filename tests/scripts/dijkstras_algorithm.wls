bd = Graph[{"a" \[DirectedEdge] "b", "a" \[DirectedEdge] "c",
   "b" \[DirectedEdge] "c", "b" \[DirectedEdge] "d",
   "c" \[DirectedEdge] "d", "d" \[DirectedEdge] "e",
   "a" \[DirectedEdge] "f", "c" \[DirectedEdge] "f",
   "e" \[DirectedEdge] "f"},
  EdgeWeight -> {7, 9, 10, 15, 11, 6, 14, 2, 9},
  VertexLabels -> "Name"];
Print[FindShortestPath[bd, "a", "e", Method -> "Dijkstra"]]
