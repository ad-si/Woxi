use super::*;

mod connected_components {
  use super::*;

  #[test]
  fn undirected_two_components() {
    let result =
      interpret("ConnectedComponents[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[4, 5]}]]")
        .unwrap();
    assert_eq!(result, "{{1, 2, 3}, {4, 5}}");
  }

  #[test]
  fn undirected_single_component() {
    let result = interpret(
      "ConnectedComponents[Graph[{UndirectedEdge[a, b], UndirectedEdge[c, d], UndirectedEdge[b, c]}]]",
    )
    .unwrap();
    assert_eq!(result, "{{a, b, c, d}}");
  }

  #[test]
  fn directed_no_cycles() {
    // Each vertex is its own SCC when there are no cycles
    let result =
      interpret("ConnectedComponents[Graph[{1 -> 2, 2 -> 3, 4 -> 5}]]")
        .unwrap();
    // Should have 5 singleton components
    assert!(result.contains("{1}"));
    assert!(result.contains("{2}"));
    assert!(result.contains("{3}"));
    assert!(result.contains("{4}"));
    assert!(result.contains("{5}"));
  }

  #[test]
  fn directed_cycle() {
    let result =
      interpret("ConnectedComponents[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]")
        .unwrap();
    // All three vertices form one SCC
    assert!(
      result.contains("1") && result.contains("2") && result.contains("3")
    );
    // Should be a single component
    assert_eq!(result.matches('{').count(), 2); // outer { + one inner {
  }

  #[test]
  fn directed_mixed() {
    let result = interpret(
      "ConnectedComponents[Graph[{1 -> 2, 2 -> 1, 3 -> 4, 4 -> 3, 1 -> 3}]]",
    )
    .unwrap();
    // {1,2} and {3,4} are separate SCCs (1->3 doesn't create a cycle between them)
    assert!(result.contains("1") && result.contains("2"));
    assert!(result.contains("3") && result.contains("4"));
  }

  #[test]
  fn complete_graph() {
    let result = interpret("ConnectedComponents[CompleteGraph[4]]").unwrap();
    assert_eq!(result, "{{1, 2, 3, 4}}");
  }

  #[test]
  fn unevaluated_non_graph() {
    assert_eq!(
      interpret("ConnectedComponents[foo]").unwrap(),
      "ConnectedComponents[foo]"
    );
  }
}

mod complete_graph {
  use super::*;

  #[test]
  fn vertices() {
    assert_eq!(
      interpret("VertexList[CompleteGraph[4]]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn edge_count_3() {
    assert_eq!(
      interpret("Length[EdgeList[CompleteGraph[3]]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn edge_count_4() {
    assert_eq!(
      interpret("Length[EdgeList[CompleteGraph[4]]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn adjacency_matrix() {
    assert_eq!(
      interpret("AdjacencyMatrix[CompleteGraph[3]]").unwrap(),
      "{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}"
    );
  }
}

mod adjacency_matrix {
  use super::*;

  #[test]
  fn directed_cycle() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}"
    );
  }

  #[test]
  fn directed_chain() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{a -> b, b -> c}]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}"
    );
  }

  #[test]
  fn single_edge() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 2}]]").unwrap(),
      "{{0, 1}, {0, 0}}"
    );
  }

  #[test]
  fn self_loop() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 1, 1 -> 2}]]").unwrap(),
      "{{1, 1}, {0, 0}}"
    );
  }
}

mod adjacency_graph_from_matrix {
  use super::*;

  #[test]
  fn undirected_symmetric() {
    assert_eq!(
      interpret("EdgeList[AdjacencyGraph[{{0, 1, 1}, {1, 0, 0}, {1, 0, 0}}]]")
        .unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3]}"
    );
    assert_eq!(
      interpret(
        "VertexList[AdjacencyGraph[{{0, 1, 1}, {1, 0, 0}, {1, 0, 0}}]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn directed_asymmetric() {
    assert_eq!(
      interpret("EdgeList[AdjacencyGraph[{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}]]")
        .unwrap(),
      "{DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[3, 1]}"
    );
    assert_eq!(
      interpret(
        "VertexList[AdjacencyGraph[{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn with_named_vertices() {
    let result = interpret(
      "EdgeList[AdjacencyGraph[{\"a\", \"b\", \"c\"}, {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}]]",
    )
    .unwrap();
    assert!(result.contains("UndirectedEdge[a, b]"));
    assert!(result.contains("UndirectedEdge[b, c]"));
  }
}

mod path_graph {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("VertexList[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("EdgeList[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 4]}"
    );
  }

  #[test]
  fn with_symbols() {
    assert_eq!(
      interpret("VertexList[PathGraph[{a, b, c}]]").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret("EdgeList[PathGraph[{a, b, c}]]").unwrap(),
      "{UndirectedEdge[a, b], UndirectedEdge[b, c]}"
    );
  }
}

mod vertex_count {
  use super::*;

  #[test]
  fn complete_graph() {
    assert_eq!(interpret("VertexCount[CompleteGraph[5]]").unwrap(), "5");
  }

  #[test]
  fn path_graph() {
    assert_eq!(interpret("VertexCount[PathGraph[{1, 2, 3}]]").unwrap(), "3");
  }
}

mod edge_count {
  use super::*;

  #[test]
  fn complete_graph() {
    assert_eq!(interpret("EdgeCount[CompleteGraph[5]]").unwrap(), "10");
  }

  #[test]
  fn path_graph() {
    assert_eq!(
      interpret("EdgeCount[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "3"
    );
  }
}

mod vertex_degree {
  use super::*;

  #[test]
  fn all_degrees() {
    assert_eq!(
      interpret("VertexDegree[CompleteGraph[4]]").unwrap(),
      "{3, 3, 3, 3}"
    );
  }

  #[test]
  fn single_vertex() {
    assert_eq!(interpret("VertexDegree[CompleteGraph[5], 1]").unwrap(), "4");
  }

  #[test]
  fn path_graph_degrees() {
    assert_eq!(
      interpret("VertexDegree[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{1, 2, 2, 1}"
    );
  }
}

mod graph_embedding {
  use super::*;

  #[test]
  fn complete_graph_3() {
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[3]]").unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn complete_graph_4() {
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[4]]").unwrap(),
      "{{-1., 0.}, {0., -1.}, {1., 0.}, {0., 1.}}"
    );
  }

  #[test]
  fn directed_graph() {
    assert_eq!(
      interpret("GraphEmbedding[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn single_edge() {
    assert_eq!(
      interpret("GraphEmbedding[Graph[{1 -> 2}]]").unwrap(),
      "{{0., -1.}, {0., 1.}}"
    );
  }

  #[test]
  fn path_graph() {
    assert_eq!(
      interpret("GraphEmbedding[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{{-1., 0.}, {0., -1.}, {1., 0.}, {0., 1.}}"
    );
  }

  #[test]
  fn non_graph_unevaluated() {
    assert_eq!(
      interpret("GraphEmbedding[42]").unwrap(),
      "GraphEmbedding[42]"
    );
  }

  #[test]
  fn with_method_argument() {
    // With explicit "CircularEmbedding" method, same result
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[3], \"CircularEmbedding\"]")
        .unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn length_matches_vertex_count() {
    assert_eq!(
      interpret("Length[GraphEmbedding[CompleteGraph[5]]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn each_coordinate_is_pair() {
    assert_eq!(
      interpret("Length[GraphEmbedding[CompleteGraph[3]][[1]]]").unwrap(),
      "2"
    );
  }
}

mod net_graph {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("NetGraph[{1, 2}, {1 -> 2}]").unwrap(),
      "NetGraph[{1, 2}, {1 -> 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[NetGraph[{1, 2}, {1 -> 2}]]").unwrap(),
      "NetGraph"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[NetGraph]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod find_maximum_flow {
  use super::*;

  #[test]
  fn simple_two_paths() {
    assert_eq!(
      interpret("FindMaximumFlow[Graph[{1 -> 2, 2 -> 3, 1 -> 3}], 1, 3]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn linear_graph() {
    assert_eq!(
      interpret("FindMaximumFlow[Graph[{1 -> 2, 2 -> 3}], 1, 3]").unwrap(),
      "1"
    );
  }

  #[test]
  fn symbolic_vertices() {
    assert_eq!(
      interpret("FindMaximumFlow[Graph[{a -> b, b -> c, a -> c}], a, c]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn no_path() {
    assert_eq!(
      interpret("FindMaximumFlow[Graph[{1 -> 2, 3 -> 4}], 1, 4]").unwrap(),
      "0"
    );
  }

  #[test]
  fn bottleneck() {
    // 1->2, 1->3, 2->4, 3->4: two paths, each capacity 1, max flow = 2
    assert_eq!(
      interpret(
        "FindMaximumFlow[Graph[{1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4}], 1, 4]"
      )
      .unwrap(),
      "2"
    );
  }
}

mod voronoi_mesh {
  use super::*;

  #[test]
  fn unevaluated_non_list() {
    assert_eq!(interpret("VoronoiMesh[x]").unwrap(), "VoronoiMesh[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[VoronoiMesh]").unwrap(), "Symbol");
  }

  #[test]
  fn single_point_returns_empty_region() {
    assert_eq!(
      interpret("VoronoiMesh[{{0, 0}}]").unwrap(),
      "EmptyRegion[2]"
    );
  }

  #[test]
  fn zero_points_returns_unevaluated() {
    assert_eq!(interpret("VoronoiMesh[{}]").unwrap(), "VoronoiMesh[{}]");
  }

  #[test]
  fn two_points_produces_mesh_region() {
    let result = interpret("VoronoiMesh[{{0, 0}, {1, 0}}]").unwrap();
    assert_eq!(result, "-Graphics-", "Got: {}", result);
  }

  #[test]
  fn three_points_produces_mesh_region() {
    let result = interpret("VoronoiMesh[{{0, 0}, {1, 0}, {0, 1}}]").unwrap();
    assert_eq!(result, "-Graphics-", "Got: {}", result);
  }

  #[test]
  fn five_points_produces_mesh_region() {
    let result =
      interpret("VoronoiMesh[{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}]")
        .unwrap();
    assert_eq!(result, "-Graphics-", "Got: {}", result);
  }

  #[test]
  fn collinear_points_produces_mesh_region() {
    let result = interpret("VoronoiMesh[{{0, 0}, {1, 0}, {2, 0}}]").unwrap();
    assert_eq!(result, "-Graphics-", "Got: {}", result);
  }

  #[test]
  fn result_is_mesh_region() {
    assert_eq!(
      interpret("Head[VoronoiMesh[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "MeshRegion"
    );
  }

  #[test]
  fn non_numeric_returns_empty_region() {
    assert_eq!(
      interpret("VoronoiMesh[{{a, b}, {c, d}}]").unwrap(),
      "EmptyRegion[2]"
    );
  }

  #[test]
  fn svg_export() {
    let result = interpret(
      "ExportString[VoronoiMesh[{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("<svg"));
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn symmetric_square() {
    // 4 corners of unit square should produce 4 cells in a MeshRegion
    let result =
      interpret("VoronoiMesh[{{0, 0}, {1, 0}, {0, 1}, {1, 1}}]").unwrap();
    assert_eq!(result, "-Graphics-", "Got: {}", result);
  }

  #[test]
  fn show_overlay_with_points() {
    // Show[VoronoiMesh[pts], Graphics[{Point[pts]}]] should render both
    let result = interpret(
      "ExportString[Show[VoronoiMesh[{{0,0},{1,0},{0,1},{1,1}}], Graphics[{Point[{{0,0},{1,0},{0,1},{1,1}}]}]], \"SVG\"]"
    ).unwrap();
    assert!(
      result.contains("<polygon"),
      "Should contain Voronoi polygons"
    );
    assert!(result.contains("<circle"), "Should contain overlay points");
  }
}

mod expression_graph {
  use super::*;

  #[test]
  fn single_atom() {
    assert_eq!(interpret("VertexList[ExpressionGraph[x]]").unwrap(), "{1}");
    assert_eq!(interpret("EdgeList[ExpressionGraph[x]]").unwrap(), "{}");
  }

  #[test]
  fn simple_function() {
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[f[x, y]]]").unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3]}"
    );
  }

  #[test]
  fn nested_function() {
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[f[x, g[y, z]]]]").unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[3, 4], UndirectedEdge[3, 5]}"
    );
  }

  #[test]
  fn plus_times() {
    assert_eq!(
      interpret("VertexList[ExpressionGraph[a + b*c]]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[a + b*c]]").unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[3, 4], UndirectedEdge[3, 5]}"
    );
  }

  #[test]
  fn list_expression() {
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[{a, b, c}]]").unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[1, 4]}"
    );
  }

  #[test]
  fn vertex_count() {
    assert_eq!(
      interpret("VertexCount[ExpressionGraph[a + b^2 + c^3 + d]]").unwrap(),
      "9"
    );
  }
}

mod graph_rendering {
  use super::*;

  #[test]
  fn undirected_graph_renders() {
    assert_eq!(
      interpret("Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 1]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn directed_graph_renders() {
    assert_eq!(
      interpret(
        "Graph[{DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[3, 1]}]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn graph_with_vertex_style() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 1]}, VertexStyle -> Orange], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("rgb(255,128,0)"));
  }

  #[test]
  fn graph_with_edge_style() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}, EdgeStyle -> Red], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("rgb(255,0,0)"));
  }

  #[test]
  fn graph_with_vertex_labels() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}, VertexLabels -> \"Name\"], \"SVG\"]"
    ).unwrap();
    assert!(result.contains(">1</text>"));
    assert!(result.contains(">2</text>"));
    assert!(result.contains(">3</text>"));
  }

  #[test]
  fn graph_with_labeled_edge() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2], Labeled[UndirectedEdge[2, 3], \"hello\"]}], \"SVG\"]"
    ).unwrap();
    assert!(result.contains(">hello</text>"));
  }

  #[test]
  fn graph_labeled_edge_label_is_black() {
    // Regression: edge labels previously inherited the edge's grey stroke
    // color instead of being rendered in black.
    let result = interpret(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, Labeled[3 <-> 1, \"hello\"]}, VertexLabels -> \"Name\"], \"SVG\"]"
    ).unwrap();
    let hello_tag = result
      .lines()
      .find(|l| l.contains(">hello</text>"))
      .expect("edge label should be rendered");
    assert!(
      hello_tag.contains("fill=\"rgb(0,0,0)\""),
      "edge label should be black, got: {hello_tag}"
    );
  }

  #[test]
  fn graph_with_diamond_shape() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2]}, VertexShapeFunction -> \"Diamond\"], \"SVG\"]"
    ).unwrap();
    // Diamond is rendered as polygon with 4 points
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn graph_with_directive_vertex_style() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2]}, VertexStyle -> Directive[Orange, EdgeForm[Orange]]], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("rgb(255,128,0)"));
  }

  #[test]
  fn graph_with_per_vertex_style_color() {
    // Style[3, Red] inside the vertex list should color that vertex red
    // without losing the edges that reference `3`.
    let result = interpret(
      "ExportString[Graph[{1, 2, Style[3, Red]}, {1 <-> 2, 2 <-> 3, 3 <-> 1}], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("fill=\"rgb(255,0,0)\""));
    // All three edges must still be drawn.
    assert_eq!(result.matches("<polyline").count(), 3);
  }

  #[test]
  fn graph_with_per_edge_style_color() {
    // Style[3 <-> 1, Green] should color that single edge green.
    let result = interpret(
      "ExportString[Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3, Style[3 <-> 1, Green]}], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("stroke=\"rgb(0,255,0)\""));
    // All three edges must still be drawn.
    assert_eq!(result.matches("<polyline").count(), 3);
  }

  #[test]
  fn graph_with_mixed_style_wrappers() {
    // Combined per-vertex and per-edge Style wrappers should coexist.
    let result = interpret(
      "ExportString[Graph[{1, 2, Style[3, Red]}, {1 <-> 2, 2 <-> 3, Style[3 <-> 1, Green]}], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("fill=\"rgb(255,0,0)\""));
    assert!(result.contains("stroke=\"rgb(0,255,0)\""));
    assert_eq!(result.matches("<polyline").count(), 3);
  }

  #[test]
  fn graph_with_styled_vertex_and_label() {
    // Label of a styled vertex should strip the Style wrapper.
    let result = interpret(
      "ExportString[Graph[{1, 2, Style[3, Red]}, {1 <-> 2, 2 <-> 3, 3 <-> 1}, VertexLabels -> \"Name\"], \"SVG\"]"
    ).unwrap();
    assert!(result.contains(">3</text>"));
    assert!(!result.contains("Style"));
  }

  #[test]
  fn complete_graph_renders() {
    assert_eq!(interpret("CompleteGraph[4]").unwrap(), "-Graphics-");
  }

  #[test]
  fn star_graph_renders() {
    assert_eq!(interpret("StarGraph[5]").unwrap(), "-Graphics-");
  }

  #[test]
  fn graph_export_string_svg() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 1]}], \"SVG\"]"
    ).unwrap();
    assert!(result.starts_with("<svg"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn directed_graph_has_arrows() {
    let result = interpret(
      "ExportString[Graph[{DirectedEdge[1, 2], DirectedEdge[2, 3]}], \"SVG\"]",
    )
    .unwrap();
    // Arrows produce polygon arrowheads
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn graph_with_vertex_size_medium() {
    // Should render without error with Medium vertex size
    assert_eq!(
      interpret("Graph[{UndirectedEdge[1, 2]}, VertexSize -> Medium]").unwrap(),
      "-Graphics-"
    );
  }

  /// Regression: vertex radii for named sizes must increase monotonically,
  /// and the spread between Tiny and Large must be visibly large (roughly
  /// matches wolframscript's ratios for a 3-vertex circular graph).
  #[test]
  fn graph_vertex_size_named_sizes_have_wide_spread() {
    let extract_radius = |code: &str| -> f64 {
      let svg = interpret(code).unwrap();
      let marker = "rx=\"";
      let start = svg.find(marker).expect("no rx attribute") + marker.len();
      let end = svg[start..].find('"').unwrap() + start;
      svg[start..end].parse::<f64>().unwrap()
    };
    let tiny = extract_radius(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, VertexSize -> Tiny], \"SVG\"]",
    );
    let small = extract_radius(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, VertexSize -> Small], \"SVG\"]",
    );
    let medium = extract_radius(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, VertexSize -> Medium], \"SVG\"]",
    );
    let large = extract_radius(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, VertexSize -> Large], \"SVG\"]",
    );
    assert!(
      tiny < small,
      "Tiny ({tiny}) should be smaller than Small ({small})"
    );
    assert!(
      small < medium,
      "Small ({small}) should be smaller than Medium ({medium})"
    );
    assert!(
      medium < large,
      "Medium ({medium}) should be smaller than Large ({large})"
    );
    // Large should be noticeably bigger than Tiny (>4x) so Table[... {Tiny,
    // Small, Medium, Large}] produces visibly different graphs.
    assert!(
      large / tiny > 4.0,
      "Large/Tiny ratio should exceed 4 (got {:.2})",
      large / tiny
    );
  }

  /// Regression: PlotLabel option on a Graph must render the title as
  /// SVG <text>. Previously the option was silently ignored.
  #[test]
  fn graph_plot_label_string() {
    let svg = interpret(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, PlotLabel -> \"My Triangle\"], \"SVG\"]"
    )
    .unwrap();
    assert!(svg.contains(">My Triangle</text>"), "SVG: {svg}");
  }

  /// PlotLabel accepts an Identifier (e.g. `PlotLabel -> Tiny` where the
  /// label is a symbolic name, as used in Table[Graph[..., PlotLabel -> s],
  /// {s, {Tiny, Small, Medium, Large}}]).
  #[test]
  fn graph_plot_label_identifier() {
    let svg = interpret(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, PlotLabel -> Tiny], \"SVG\"]"
    )
    .unwrap();
    assert!(svg.contains(">Tiny</text>"), "SVG: {svg}");
  }

  /// Style directives inside PlotLabel (color, font size, Bold, Italic)
  /// must flow through to the rendered <text> element, overriding defaults.
  #[test]
  fn graph_plot_label_styled() {
    let svg = interpret(
      "ExportString[Graph[{1 <-> 2}, PlotLabel -> Style[\"Red Title\", Red, 20, Italic]], \"SVG\"]"
    )
    .unwrap();
    assert!(svg.contains(">Red Title</text>"), "SVG: {svg}");
    assert!(
      svg.contains("fill=\"rgb(255,0,0)\""),
      "expected red fill; SVG: {svg}"
    );
    assert!(
      svg.contains("font-size=\"20\""),
      "expected font-size 20; SVG: {svg}"
    );
    assert!(
      svg.contains("font-style=\"italic\""),
      "expected italic; SVG: {svg}"
    );
  }

  /// PlotLabel -> None must suppress the label entirely (no <text> added
  /// beyond whatever other options may add).
  #[test]
  fn graph_plot_label_none_renders_no_label() {
    let svg = interpret(
      "ExportString[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}, PlotLabel -> None], \"SVG\"]"
    )
    .unwrap();
    assert!(
      !svg.contains("<text"),
      "SVG should have no <text> element: {svg}"
    );
  }

  #[test]
  fn graph_preserves_vertex_list() {
    assert_eq!(
      interpret(
        "VertexList[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn graph_preserves_edge_list() {
    assert_eq!(
      interpret(
        "EdgeList[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}]]"
      )
      .unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"
    );
  }

  #[test]
  fn graph_with_square_shape() {
    let result = interpret(
      "ExportString[Graph[{UndirectedEdge[1, 2]}, VertexShapeFunction -> \"Square\"], \"SVG\"]"
    ).unwrap();
    assert!(result.starts_with("<svg"));
  }

  #[test]
  fn directed_self_loop() {
    let result = interpret(
      "ExportString[Graph[{1, 2, 3}, {UndirectedEdge[1, 2], UndirectedEdge[2, 3], DirectedEdge[2, 2]}], \"SVG\"]"
    ).unwrap();
    assert!(result.starts_with("<svg"));
    // Self-loop produces an arrowhead polygon
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn undirected_self_loop() {
    let result = interpret(
      "ExportString[Graph[{1, 2}, {UndirectedEdge[1, 2], UndirectedEdge[1, 1]}], \"SVG\"]"
    ).unwrap();
    assert!(result.starts_with("<svg"));
    // Should have 2 polylines: one regular edge + one self-loop
    assert_eq!(result.matches("<polyline").count(), 2);
  }
}

mod group_orbits {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("GroupOrbits[x, y]").unwrap(), "GroupOrbits[x, y]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GroupOrbits]").unwrap(), "Symbol");
  }
}

mod two_way_rule {
  use super::*;

  #[test]
  fn operator_parses_as_two_way_rule() {
    // Matches Wolfram: a <-> b has Head TwoWayRule (not UndirectedEdge).
    assert_eq!(interpret("FullForm[a <-> b]").unwrap(), "TwoWayRule[a, b]");
    assert_eq!(interpret("Head[a <-> b]").unwrap(), "TwoWayRule");
  }

  #[test]
  fn roundtrip_display() {
    assert_eq!(interpret("a <-> b").unwrap(), "a <-> b");
    assert_eq!(interpret("TwoWayRule[a, b]").unwrap(), "a <-> b");
  }

  #[test]
  fn precedence_tighter_than_rule() {
    // `<->` binds tighter than `->`: a <-> b -> c == Rule[TwoWayRule[a, b], c]
    assert_eq!(
      interpret("FullForm[a <-> b -> c]").unwrap(),
      "Rule[TwoWayRule[a, b], c]"
    );
  }

  #[test]
  fn precedence_looser_than_arithmetic() {
    // `<->` binds looser than `+`: a + b <-> c + d == TwoWayRule[a+b, c+d]
    assert_eq!(
      interpret("FullForm[a + b <-> c + d]").unwrap(),
      "TwoWayRule[Plus[a, b], Plus[c, d]]"
    );
  }

  #[test]
  fn inside_list() {
    // List elements containing `<->` must not be mis-parsed as ReplacementRule.
    assert_eq!(
      interpret("{a <-> b, c <-> d}").unwrap(),
      "{a <-> b, c <-> d}"
    );
  }

  #[test]
  fn inside_function_args() {
    assert_eq!(interpret("f[a <-> b]").unwrap(), "f[a <-> b]");
    // Mixed with a real rule argument — both should parse distinctly.
    assert_eq!(
      interpret("f[a <-> b, x -> y]").unwrap(),
      "f[a <-> b, x -> y]"
    );
  }

  #[test]
  fn graph_accepts_two_way_rule_edges() {
    // Graph[] should treat TwoWayRule edges as undirected edges.
    assert_eq!(
      interpret("VertexCount[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("EdgeCount[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 1}]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn graph_two_way_rule_connected_components() {
    assert_eq!(
      interpret("ConnectedComponents[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}]]")
        .unwrap(),
      "{{1, 2, 3}, {4, 5}}"
    );
  }

  #[test]
  fn graph_two_way_rule_with_self_loop() {
    // Regression for the user's `Graph[{"A" <-> "B", "B" <-> "B"}]` example
    // — parser must accept self-loops with `<->` without failing.
    assert_eq!(
      interpret("VertexCount[Graph[{\"A\" <-> \"B\", \"B\" <-> \"B\"}]]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn graph_mixed_directed_and_two_way_rule() {
    // Regression: mixing `->` (Rule → DirectedEdge) with `<->` in the
    // same edge list must work. Wolfram accepts this.
    assert_eq!(
      interpret("VertexList[Graph[{a -> b, b -> a, b <-> b}]]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("VertexCount[Graph[{a -> b, b -> a, b <-> b}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("EdgeCount[Graph[{a -> b, b -> a, b <-> b}]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("EdgeList[Graph[{a -> b, b -> a, b <-> b}]]").unwrap(),
      "{DirectedEdge[a, b], DirectedEdge[b, a], UndirectedEdge[b, b]}"
    );
  }

  #[test]
  fn graph_mixed_rule_and_undirected_edge() {
    // Same mixed case but with an explicit UndirectedEdge[..] alongside a Rule.
    assert_eq!(
      interpret("EdgeList[Graph[{1 -> 2, UndirectedEdge[2, 3]}]]").unwrap(),
      "{DirectedEdge[1, 2], UndirectedEdge[2, 3]}"
    );
  }

  #[test]
  fn antiparallel_edges_render_as_distinct_curves() {
    // Regression: Graph[{a -> b, b -> a}] must render two visually
    // distinct arrow curves, not two arrows overlapping on a single
    // straight line between the vertices.
    let result =
      interpret("ExportString[Graph[{a -> b, b -> a}], \"SVG\"]").unwrap();
    // Two polylines for the two edges…
    assert_eq!(result.matches("<polyline").count(), 2);
    // …and two arrowhead polygons (one per directed edge).
    assert_eq!(result.matches("<polygon").count(), 2);
    // The two polylines must not be identical (that was the bug —
    // both edges were drawn as the same horizontal segment).
    let polylines: Vec<&str> = result
      .match_indices("<polyline")
      .map(|(i, _)| {
        let end = result[i..].find("/>").unwrap_or(0) + i;
        &result[i..end]
      })
      .collect();
    assert_ne!(polylines[0], polylines[1]);
  }

  #[test]
  fn mixed_antiparallel_plus_self_loop_renders_three_edges() {
    // The original report: Graph[{a -> b, b -> a, b <-> b}] must have
    // two separate edges between a and b (curved) plus a self-loop on b.
    // Expect 3 polylines total (2 curves + 1 loop) and 2 arrowheads
    // (one per directed antiparallel edge; the self-loop is undirected).
    let result =
      interpret("ExportString[Graph[{a -> b, b -> a, b <-> b}], \"SVG\"]")
        .unwrap();
    assert_eq!(result.matches("<polyline").count(), 3);
    assert_eq!(result.matches("<polygon").count(), 2);
  }

  #[test]
  fn labeled_two_way_rule_edge() {
    // Labeled wrapper around `<->` should still be recognized as an edge.
    assert_eq!(
      interpret("EdgeCount[Graph[{Labeled[a <-> b, \"e1\"]}]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn short_arrow_edges_get_proportionally_smaller_arrowheads() {
    // Regression: multi-component cluster graphs pack many short edges
    // into small cells. The arrowhead must scale down with the edge so
    // it doesn't swallow the whole line. For each directed edge we
    // verify that the arrowhead's triangle is no larger than ~half the
    // edge's polyline length in pixel space.
    let svg = interpret(
      "ExportString[Graph[Table[i -> Mod[i^2, 74], {i, 100}]], \"SVG\"]",
    )
    .unwrap();

    // Extract (polyline points, polygon points) pairs — in graph
    // rendering each directed edge produces one polyline followed
    // immediately by one arrowhead polygon.
    let polyline_re = "<polyline points=\"";
    let polygon_re = "<polygon points=\"";

    let mut idx = 0;
    let mut pairs_checked = 0;
    while let Some(p_off) = svg[idx..].find(polyline_re) {
      let p_start = idx + p_off + polyline_re.len();
      let p_end = svg[p_start..].find('"').map(|e| p_start + e).unwrap();
      let polyline_pts_str = &svg[p_start..p_end];

      // Find the next polygon after this polyline.
      let after = p_end;
      let Some(g_off) = svg[after..].find(polygon_re) else {
        break;
      };
      let g_start = after + g_off + polygon_re.len();
      let g_end = svg[g_start..].find('"').map(|e| g_start + e).unwrap();
      let polygon_pts_str = &svg[g_start..g_end];

      idx = g_end;

      // Parse points like "x,y x,y ..." into Vec<(f64,f64)>.
      let parse = |s: &str| -> Vec<(f64, f64)> {
        s.split_whitespace()
          .filter_map(|tok| {
            let (a, b) = tok.split_once(',')?;
            Some((a.parse().ok()?, b.parse().ok()?))
          })
          .collect()
      };
      let pl = parse(polyline_pts_str);
      let pg = parse(polygon_pts_str);
      if pl.len() < 2 || pg.len() != 3 {
        continue;
      }

      // Polyline length in pixels, and bounding-box diagonal.
      let mut line_len = 0.0_f64;
      let (mut minx, mut maxx) = (f64::INFINITY, f64::NEG_INFINITY);
      let (mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY);
      for &(x, y) in &pl {
        if x < minx {
          minx = x;
        }
        if x > maxx {
          maxx = x;
        }
        if y < miny {
          miny = y;
        }
        if y > maxy {
          maxy = y;
        }
      }
      for w in pl.windows(2) {
        line_len +=
          ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
      }
      let bbox_diag = ((maxx - minx).powi(2) + (maxy - miny).powi(2)).sqrt();

      // Arrowhead length = distance from tip to midpoint of the base.
      // The tip is vertex 0 of the polygon (emitted first in graphics.rs).
      let tip = pg[0];
      let base_mid = ((pg[1].0 + pg[2].0) / 2.0, (pg[1].1 + pg[2].1) / 2.0);
      let head_len =
        ((tip.0 - base_mid.0).powi(2) + (tip.1 - base_mid.1).powi(2)).sqrt();

      // Skip pathologically collapsed edges (e.g. FR layouts sometimes
      // put two nodes on top of each other, producing a line of length
      // < 1 px). The arrow is invisible anyway.
      if line_len < 2.0 {
        continue;
      }

      // Head must be small compared to both the path length and the
      // shape's bbox diagonal (the latter catches tight self-loops).
      let shape_size = line_len.min(bbox_diag).max(1.0);
      assert!(
        head_len <= shape_size * 0.6 + 0.5,
        "arrowhead too large: head_len={head_len:.2} shape_size={shape_size:.2}"
      );
      pairs_checked += 1;
    }
    assert!(
      pairs_checked >= 50,
      "expected to find many arrow edges, got {pairs_checked}"
    );
  }

  #[test]
  fn multi_component_graph_renders_separated_clusters() {
    // Regression: Graph[Table[i -> Mod[i^2, 74], {i, 100}]] has 8
    // weakly-connected components (sizes 34, 32, 12, 10, 5, 5, 2, 1).
    // A plain circular layout would place all 101 vertices on a single
    // circle and hide the cluster structure, so we lay out each
    // component separately and pack them into a grid. Verify that the
    // SVG contains all 101 vertices and that their centers are not all
    // on a common circle (which would indicate the old circular layout
    // is still being used for multi-component graphs).
    let svg = interpret(
      "ExportString[Graph[Table[i -> Mod[i^2, 74], {i, 100}]], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));

    // Collect vertex centers from the rendered <ellipse cx="..." cy="..." ...> tags.
    let centers: Vec<(f64, f64)> = svg
      .match_indices("<ellipse")
      .filter_map(|(i, _)| {
        let rest = &svg[i..];
        let cx_idx = rest.find("cx=\"")? + 4;
        let cx_end = rest[cx_idx..].find('"')? + cx_idx;
        let cy_idx = rest.find("cy=\"")? + 4;
        let cy_end = rest[cy_idx..].find('"')? + cy_idx;
        let cx: f64 = rest[cx_idx..cx_end].parse().ok()?;
        let cy: f64 = rest[cy_idx..cy_end].parse().ok()?;
        Some((cx, cy))
      })
      .collect();
    assert_eq!(
      centers.len(),
      101,
      "expected 101 rendered vertices, got {}",
      centers.len()
    );

    // If every vertex were on a common circle its distance to the bbox
    // centroid would be nearly constant. Check that the stddev of those
    // distances is substantially larger than 5% of the mean — i.e. the
    // layout is clearly not a single circle.
    let cx0: f64 =
      centers.iter().map(|p| p.0).sum::<f64>() / centers.len() as f64;
    let cy0: f64 =
      centers.iter().map(|p| p.1).sum::<f64>() / centers.len() as f64;
    let dists: Vec<f64> = centers
      .iter()
      .map(|&(x, y)| ((x - cx0).powi(2) + (y - cy0).powi(2)).sqrt())
      .collect();
    let mean = dists.iter().sum::<f64>() / dists.len() as f64;
    let var = dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
      / dists.len() as f64;
    let stddev = var.sqrt();
    assert!(
      stddev / mean > 0.15,
      "vertices look like a single circle (stddev/mean = {:.3})",
      stddev / mean
    );
  }
}
