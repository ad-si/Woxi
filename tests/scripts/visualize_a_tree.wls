(* Raw ExportString[…, "SVG"] bytes can never match between engines:
   wolframscript emits cairo-rendered SVG (font glyph outlines, pt units)
   while Woxi emits its own primitive-based SVG. Assert stable projections
   instead: the edge structure and that both renderers produce SVG. *)

edges = {
  1 \[DirectedEdge] 2,
  1 \[DirectedEdge] 3,
  2 \[DirectedEdge] 4,
  2 \[DirectedEdge] 5,
  3 \[DirectedEdge] 6,
  4 \[DirectedEdge] 7
};

Print[edges];

treeSvg = ExportString[
  TreeGraph[edges, GraphStyle -> "VintageDiagram"],
  "SVG"
];
Print[StringQ[treeSvg] && StringContainsQ[treeSvg, "<svg"]];

formSvg = ExportString[
  TreeForm[Defer@
    TreeGraph[{1 \[DirectedEdge] 2, 1 \[DirectedEdge] 3, 2 \[DirectedEdge] 4, 2 \[DirectedEdge] 5,
    3 \[DirectedEdge] 6,  4 \[DirectedEdge] 7}, VertexLabels -> "Name"]],
  "SVG"
];
Print[StringQ[formSvg] && StringContainsQ[formSvg, "<svg"]];
