edges = {
  1 \[DirectedEdge] 2,
  1 \[DirectedEdge] 3,
  2 \[DirectedEdge] 4,
  2 \[DirectedEdge] 5,
  3 \[DirectedEdge] 6,
  4 \[DirectedEdge] 7
};

Print[ExportString[
  TreeGraph[edges, GraphStyle -> "VintageDiagram"],
  "SVG"
]]

Print[ExportString[
  TreeForm[Defer@
    TreeGraph[{1 \[DirectedEdge] 2, 1 \[DirectedEdge] 3, 2 \[DirectedEdge] 4, 2 \[DirectedEdge] 5,
    3 \[DirectedEdge] 6,  4 \[DirectedEdge] 7}, VertexLabels -> "Name"]],
  "SVG"
]]
