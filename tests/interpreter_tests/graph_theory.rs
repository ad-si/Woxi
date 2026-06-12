use super::*;

mod connected_components {
  use super::*;

  #[test]
  fn undirected_two_components() {
    let result =
      interpret("ConnectedComponents[Graph[{1  2, 2  3, 4  5}]]").unwrap();
    assert_eq!(result, "{{1, 2, 3}, {4, 5}}");
  }

  #[test]
  fn undirected_single_component() {
    let result =
      interpret("ConnectedComponents[Graph[{a  b, c  d, b  c}]]").unwrap();
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

mod connected_graph_q {
  use super::*;

  // A directed path is not strongly connected.
  #[test]
  fn directed_path_not_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1 -> 2, 2 -> 3}]]").unwrap(),
      "False"
    );
  }

  // A directed cycle is strongly connected.
  #[test]
  fn directed_cycle_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn directed_bidirectional_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1 -> 2, 2 -> 1}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn undirected_path_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1 <-> 2, 2 <-> 3}]]").unwrap(),
      "True"
    );
  }

  // Two disconnected undirected components.
  #[test]
  fn undirected_two_components_not_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}]]").unwrap(),
      "False"
    );
  }

  // An isolated vertex makes the graph disconnected.
  #[test]
  fn isolated_vertex_not_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1, 2, 3}, {UndirectedEdge[1, 2]}]]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn complete_graph_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[CompleteGraph[4]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn single_vertex_connected() {
    assert_eq!(
      interpret("ConnectedGraphQ[Graph[{1}, {}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn empty_graph_connected() {
    assert_eq!(interpret("ConnectedGraphQ[Graph[{}, {}]]").unwrap(), "True");
  }

  // Non-graph arguments yield False (not unevaluated).
  #[test]
  fn non_graph_is_false() {
    assert_eq!(interpret("ConnectedGraphQ[5]").unwrap(), "False");
    assert_eq!(interpret(r#"ConnectedGraphQ["hello"]"#).unwrap(), "False");
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

  #[test]
  fn with_vertex_size_option() {
    // CompleteGraph now accepts trailing Rule options.
    assert_eq!(
      interpret("Head[CompleteGraph[4, VertexSize -> Small]]").unwrap(),
      "Graph"
    );
    assert_eq!(
      interpret("Length[VertexList[CompleteGraph[4, VertexSize -> Small]]]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Length[EdgeList[CompleteGraph[4, VertexSize -> Small]]]")
        .unwrap(),
      "6"
    );
  }

  #[test]
  fn cycle_graph_with_options() {
    assert_eq!(
      interpret("Head[CycleGraph[5, VertexSize -> {1 -> Medium}]]").unwrap(),
      "Graph"
    );
    assert_eq!(
      interpret("Length[EdgeList[CycleGraph[5, VertexSize -> {1 -> Medium}]]]")
        .unwrap(),
      "5"
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

mod incidence_matrix {
  use super::*;

  #[test]
  fn undirected_complete() {
    // Rows = vertices, columns = edges; both endpoints get +1.
    assert_eq!(
      interpret("IncidenceMatrix[CompleteGraph[3]]").unwrap(),
      "{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}"
    );
  }

  #[test]
  fn undirected_path() {
    assert_eq!(
      // Normal[...] makes the comparison representation-agnostic: Woxi already
      // returns a dense matrix, and wolframscript's SparseArray collapses to
      // the same dense list, so the conformance check holds for both.
      interpret("Normal[IncidenceMatrix[PathGraph[{1, 2, 3, 4}]]]").unwrap(),
      "{{1, 0, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}}"
    );
  }

  #[test]
  fn directed_cycle() {
    // Directed edge: source -1, target +1.
    assert_eq!(
      interpret(
        "Normal[IncidenceMatrix[Graph[{1, 2, 3}, {1 -> 2, 2 -> 3, 3 -> 1}]]]"
      )
      .unwrap(),
      "{{-1, 0, 1}, {1, -1, 0}, {0, 1, -1}}"
    );
  }

  #[test]
  fn single_directed_edge() {
    assert_eq!(
      interpret("Normal[IncidenceMatrix[Graph[{1, 2, 3}, {1 -> 2}]]]").unwrap(),
      "{{-1}, {1}, {0}}"
    );
  }

  #[test]
  fn directed_self_loop() {
    // A directed self-loop yields -2 (Wolfram convention).
    assert_eq!(
      interpret("Normal[IncidenceMatrix[Graph[{1, 2}, {1 -> 1, 1 -> 2}]]]")
        .unwrap(),
      "{{-2, -1}, {0, 1}}"
    );
  }

  #[test]
  fn undirected_self_loop() {
    // An undirected self-loop yields 2.
    assert_eq!(
      interpret("Normal[IncidenceMatrix[Graph[{1, 2}, {1 <-> 1, 1 <-> 2}]]]")
        .unwrap(),
      "{{2, 1}, {0, 1}}"
    );
  }
}

mod kirchhoff_matrix {
  use super::*;

  #[test]
  fn complete_graph() {
    // L = D - A; the complete graph K_n has diagonal n-1 and -1 off-diagonal.
    assert_eq!(
      interpret("KirchhoffMatrix[CompleteGraph[3]]").unwrap(),
      "{{2, -1, -1}, {-1, 2, -1}, {-1, -1, 2}}"
    );
    assert_eq!(
      interpret("KirchhoffMatrix[CompleteGraph[4]]").unwrap(),
      "{{3, -1, -1, -1}, {-1, 3, -1, -1}, \
       {-1, -1, 3, -1}, {-1, -1, -1, 3}}"
    );
  }

  #[test]
  fn undirected_path() {
    assert_eq!(
      // Normal[...] reconciles Woxi's dense matrix with wolframscript's
      // SparseArray result without changing the values.
      interpret(
        "Normal[KirchhoffMatrix[Graph[{1, 2, 3, 4}, {1 <-> 2, 2 <-> 3, 3 <-> 4}]]]"
      )
      .unwrap(),
      "{{1, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 1}}"
    );
  }

  #[test]
  fn directed_chain() {
    // Directed edges count toward both the source's out-degree and the
    // target's in-degree on the diagonal, with -A on the off-diagonal.
    assert_eq!(
      interpret("Normal[KirchhoffMatrix[Graph[{1, 2, 3}, {1 -> 2, 2 -> 3}]]]")
        .unwrap(),
      "{{1, -1, 0}, {0, 2, -1}, {0, 0, 1}}"
    );
  }

  #[test]
  fn isolated_vertices() {
    assert_eq!(
      interpret("Normal[KirchhoffMatrix[Graph[{1, 2, 3, 4}, {1 <-> 2}]]]")
        .unwrap(),
      "{{1, -1, 0, 0}, {-1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn directed_self_loop_ignored() {
    // Self-loops contribute nothing to the Kirchhoff matrix.
    assert_eq!(
      interpret("Normal[KirchhoffMatrix[Graph[{1, 2}, {1 -> 1, 1 -> 2}]]]")
        .unwrap(),
      "{{1, -1}, {0, 1}}"
    );
  }

  #[test]
  fn undirected_self_loop_ignored() {
    assert_eq!(
      interpret("Normal[KirchhoffMatrix[Graph[{1, 2}, {1 <-> 1, 1 <-> 2}]]]")
        .unwrap(),
      "{{1, -1}, {-1, 1}}"
    );
  }

  #[test]
  fn parallel_edges_collapsed() {
    // Parallel edges collapse to a simple graph for the Kirchhoff matrix.
    assert_eq!(
      interpret(
        "Normal[KirchhoffMatrix[Graph[{1, 2, 3}, {1 <-> 2, 1 <-> 2, 2 <-> 3}]]]"
      )
      .unwrap(),
      "{{1, -1, 0}, {-1, 2, -1}, {0, -1, 1}}"
    );
  }

  #[test]
  fn non_graph_stays_symbolic() {
    assert_eq!(
      interpret("KirchhoffMatrix[5]").unwrap(),
      "KirchhoffMatrix[5]"
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
      "{1  2, 1  3}"
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
      "{1  2, 2  3, 3  1}"
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
    assert!(result.contains("a  b"));
    assert!(result.contains("b  c"));
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
      "{1  2, 2  3, 3  4}"
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
      "{a  b, b  c}"
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
      "{1  2, 1  3}"
    );
  }

  #[test]
  fn nested_function() {
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[f[x, g[y, z]]]]").unwrap(),
      "{1  2, 1  3, 3  4, 3  5}"
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
      "{1  2, 1  3, 3  4, 3  5}"
    );
  }

  #[test]
  fn list_expression() {
    assert_eq!(
      interpret("EdgeList[ExpressionGraph[{a, b, c}]]").unwrap(),
      "{1  2, 1  3, 1  4}"
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
    // Graph summarises as `<vertex_count>, <edge_count>`.
    assert_eq!(
      interpret("Graph[{1  2, 2  3, 3  1}]").unwrap(),
      "Graph[<3>, <3>]"
    );
  }

  #[test]
  fn directed_graph_renders() {
    assert_eq!(
      interpret("Graph[{1  2, 2  3, 3  1}]").unwrap(),
      "Graph[<3>, <3>]"
    );
  }

  #[test]
  fn graph_with_vertex_style() {
    let result = interpret(
      "ExportString[Graph[{1  2, 2  3, 3  1}, VertexStyle -> Orange], \"SVG\"]"
    ).unwrap();
    assert!(result.contains("rgb(255,128,0)"));
  }

  #[test]
  fn graph_with_edge_style() {
    let result = interpret(
      "ExportString[Graph[{1  2, 2  3}, EdgeStyle -> Red], \"SVG\"]",
    )
    .unwrap();
    assert!(result.contains("rgb(255,0,0)"));
  }

  #[test]
  fn graph_with_vertex_labels() {
    let result = interpret(
      "ExportString[Graph[{1  2, 2  3}, VertexLabels -> \"Name\"], \"SVG\"]",
    )
    .unwrap();
    assert!(result.contains(">1</text>"));
    assert!(result.contains(">2</text>"));
    assert!(result.contains(">3</text>"));
  }

  #[test]
  fn graph_with_labeled_edge() {
    let result = interpret(
      "ExportString[Graph[{1  2, Labeled[2  3, \"hello\"]}], \"SVG\"]",
    )
    .unwrap();
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
      "ExportString[Graph[{1  2}, VertexShapeFunction -> \"Diamond\"], \"SVG\"]"
    ).unwrap();
    // Diamond is rendered as polygon with 4 points
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn graph_with_directive_vertex_style() {
    let result = interpret(
      "ExportString[Graph[{1  2}, VertexStyle -> Directive[Orange, EdgeForm[Orange]]], \"SVG\"]"
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
    assert_eq!(interpret("CompleteGraph[4]").unwrap(), "Graph[<4>, <6>]");
  }

  #[test]
  fn star_graph_renders() {
    assert_eq!(interpret("StarGraph[5]").unwrap(), "Graph[<5>, <4>]");
  }

  #[test]
  fn grid_graph_vertex_count() {
    assert_eq!(
      interpret("VertexCount[GridGraph[{10, 10}]]").unwrap(),
      "100"
    );
    assert_eq!(interpret("VertexCount[GridGraph[{2, 3}]]").unwrap(), "6");
    assert_eq!(interpret("VertexCount[GridGraph[{3, 2}]]").unwrap(), "6");
    assert_eq!(interpret("VertexCount[GridGraph[{1, 1}]]").unwrap(), "1");
  }

  #[test]
  fn grid_graph_edge_count() {
    // {m, n}: horizontal edges = (m-1)*n, vertical edges = m*(n-1)
    assert_eq!(interpret("EdgeCount[GridGraph[{2, 3}]]").unwrap(), "7");
    assert_eq!(interpret("EdgeCount[GridGraph[{3, 2}]]").unwrap(), "7");
    assert_eq!(interpret("EdgeCount[GridGraph[{10, 10}]]").unwrap(), "180");
    assert_eq!(interpret("EdgeCount[GridGraph[{1, 1}]]").unwrap(), "0");
  }

  #[test]
  fn grid_graph_edge_list_2x3() {
    // GridGraph[{m, n}]: m columns, n rows, row-major numbering.
    // For each v: if not in last column emit v—(v+1); if not in last row emit v—(v+m).
    assert_eq!(
      interpret("EdgeList[GridGraph[{2, 3}]]").unwrap(),
      format!(
        "{{1 {ue} 2, 1 {ue} 3, 2 {ue} 4, 3 {ue} 4, 3 {ue} 5, 4 {ue} 6, 5 {ue} 6}}",
        ue = "\u{f3d4}"
      )
    );
  }

  #[test]
  fn grid_graph_edge_list_3x2() {
    assert_eq!(
      interpret("EdgeList[GridGraph[{3, 2}]]").unwrap(),
      format!(
        "{{1 {ue} 2, 1 {ue} 4, 2 {ue} 3, 2 {ue} 5, 3 {ue} 6, 4 {ue} 5, 5 {ue} 6}}",
        ue = "\u{f3d4}"
      )
    );
  }

  #[test]
  fn grid_graph_renders() {
    assert_eq!(interpret("GridGraph[{3, 3}]").unwrap(), "Graph[<9>, <12>]");
  }

  #[test]
  fn graph_export_string_svg() {
    let result =
      interpret("ExportString[Graph[{1  2, 2  3, 3  1}], \"SVG\"]").unwrap();
    assert!(result.starts_with("<svg"));
    assert!(result.contains("</svg>"));
  }

  #[test]
  fn directed_graph_has_arrows() {
    let result =
      interpret("ExportString[Graph[{1  2, 2  3}], \"SVG\"]").unwrap();
    // Arrows produce polygon arrowheads
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn graph_with_vertex_size_medium() {
    // Should render without error with Medium vertex size
    assert_eq!(
      interpret("Graph[{1  2}, VertexSize -> Medium]").unwrap(),
      "Graph[<2>, <1>]"
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
      interpret("VertexList[Graph[{1  2, 2  3}]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn graph_preserves_edge_list() {
    assert_eq!(
      interpret("EdgeList[Graph[{1  2, 2  3}]]").unwrap(),
      "{1  2, 2  3}"
    );
  }

  #[test]
  fn graph_with_square_shape() {
    let result = interpret(
      "ExportString[Graph[{1  2}, VertexShapeFunction -> \"Square\"], \"SVG\"]"
    ).unwrap();
    assert!(result.starts_with("<svg"));
  }

  #[test]
  fn directed_self_loop() {
    let result = interpret(
      "ExportString[Graph[{1, 2, 3}, {1  2, 2  3, 2  2}], \"SVG\"]",
    )
    .unwrap();
    assert!(result.starts_with("<svg"));
    // Self-loop produces an arrowhead polygon
    assert!(result.contains("<polygon"));
  }

  #[test]
  fn undirected_self_loop() {
    let result =
      interpret("ExportString[Graph[{1, 2}, {1  2, 1  1}], \"SVG\"]")
        .unwrap();
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
    assert_eq!(interpret("FullForm[a <-> b]").unwrap(), "FullForm[a <-> b]");
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
      "FullForm[a <-> b -> c]"
    );
  }

  #[test]
  fn precedence_looser_than_arithmetic() {
    // `<->` binds looser than `+`: a + b <-> c + d == TwoWayRule[a+b, c+d]
    assert_eq!(
      interpret("FullForm[a + b <-> c + d]").unwrap(),
      "FullForm[a + b <-> c + d]"
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
      "{a  b, b  a, b  b}"
    );
  }

  #[test]
  fn graph_mixed_rule_and_undirected_edge() {
    // Same mixed case but with an explicit UndirectedEdge[..] alongside a Rule.
    assert_eq!(
      interpret("EdgeList[Graph[{1 -> 2, 2  3}]]").unwrap(),
      "{1  2, 2  3}"
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

mod random_graph {
  use super::*;

  #[test]
  fn correct_vertex_count() {
    assert_eq!(
      interpret("Length[VertexList[RandomGraph[{5, 6}]]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn correct_edge_count() {
    assert_eq!(
      interpret("Length[EdgeList[RandomGraph[{5, 6}]]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn head_is_graph() {
    assert_eq!(interpret("Head[RandomGraph[{5, 6}]]").unwrap(), "Graph");
  }

  #[test]
  fn edges_are_unique() {
    assert_eq!(
      interpret("Length[DeleteDuplicates[EdgeList[RandomGraph[{8, 12}]]]]")
        .unwrap(),
      "12"
    );
  }

  #[test]
  fn complete_graph_when_m_equals_max() {
    // K_4 has 6 edges; RandomGraph[{4, 6}] should yield exactly that set.
    assert_eq!(
      interpret("Length[EdgeList[RandomGraph[{4, 6}]]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn too_many_edges_returns_unevaluated() {
    // K_4 has 6 edges, asking for 100 is impossible.
    assert_eq!(
      interpret("RandomGraph[{4, 100}]").unwrap(),
      "RandomGraph[{4, 100}]"
    );
  }

  #[test]
  fn k_variant_returns_list() {
    assert_eq!(interpret("Length[RandomGraph[{5, 4}, 3]]").unwrap(), "3");
    assert_eq!(
      interpret("Head /@ RandomGraph[{5, 4}, 3]").unwrap(),
      "{Graph, Graph, Graph}"
    );
  }

  #[test]
  fn with_options() {
    // Audit case: trailing Rule options pass through as Graph options.
    assert_eq!(
      interpret(
        "Head[RandomGraph[{10, 20}, VertexLabels -> Placed[Automatic, Center], \
         VertexSize -> 0.75]]"
      )
      .unwrap(),
      "Graph"
    );
    assert_eq!(
      interpret(
        "Length[VertexList[RandomGraph[{10, 20}, \
         VertexLabels -> Placed[Automatic, Center], VertexSize -> 0.75]]]"
      )
      .unwrap(),
      "10"
    );
    assert_eq!(
      interpret(
        "Length[EdgeList[RandomGraph[{10, 20}, \
         VertexLabels -> Placed[Automatic, Center], VertexSize -> 0.75]]]"
      )
      .unwrap(),
      "20"
    );
  }

  #[test]
  fn bernoulli_graph_distribution_returns_graph() {
    // RandomGraph[BernoulliGraphDistribution[n, p]] is an Erdős–Rényi
    // G(n, p) sample: each of the n·(n−1)/2 possible edges is included
    // independently with probability p. The result is a `Graph` with
    // exactly n vertices.
    assert_eq!(
      interpret("Head[RandomGraph[BernoulliGraphDistribution[6, 0.4]]]")
        .unwrap(),
      "Graph"
    );
    assert_eq!(
      interpret(
        "Length[VertexList[RandomGraph[BernoulliGraphDistribution[6, 0.4]]]]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn bernoulli_graph_distribution_edge_count_in_range() {
    // p=1 forces all edges; p=0 forbids them.
    assert_eq!(
      interpret(
        "Length[EdgeList[RandomGraph[BernoulliGraphDistribution[5, 1]]]]"
      )
      .unwrap(),
      "10"
    );
    assert_eq!(
      interpret(
        "Length[EdgeList[RandomGraph[BernoulliGraphDistribution[5, 0]]]]"
      )
      .unwrap(),
      "0"
    );
  }
}

mod graph_q {
  use super::*;

  #[test]
  fn true_for_edge_list_graph() {
    assert_eq!(interpret("GraphQ[Graph[{1->2, 2->3}]]").unwrap(), "True");
  }

  #[test]
  fn true_for_complete_graph() {
    assert_eq!(interpret("GraphQ[CompleteGraph[4]]").unwrap(), "True");
  }

  #[test]
  fn true_for_path_graph() {
    assert_eq!(interpret("GraphQ[PathGraph[{1, 2, 3}]]").unwrap(), "True");
  }

  #[test]
  fn true_for_vertices_and_rule_edges() {
    assert_eq!(
      interpret("GraphQ[Graph[{1, 2, 3}, {1 -> 2}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn true_for_empty_edge_list() {
    assert_eq!(interpret("GraphQ[Graph[{1, 2}, {}]]").unwrap(), "True");
  }

  #[test]
  fn false_for_number() {
    assert_eq!(interpret("GraphQ[5]").unwrap(), "False");
  }

  #[test]
  fn false_for_string() {
    assert_eq!(interpret(r#"GraphQ["abc"]"#).unwrap(), "False");
  }

  #[test]
  fn false_for_bare_edge_list() {
    assert_eq!(interpret("GraphQ[{1 -> 2}]").unwrap(), "False");
  }

  #[test]
  fn false_for_invalid_edge() {
    assert_eq!(
      interpret(r#"GraphQ[Graph[{1, 2, 3}, {1 -> 2, "x"}]]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn false_for_non_edge_in_edge_list() {
    assert_eq!(
      interpret("GraphQ[Graph[{1, 2, 3}, {1, 2}]]").unwrap(),
      "False"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn directed_edge() {
    assert_case(r#"DirectedEdge[x, y, z]"#, r#"DirectedEdge[x, y, z]"#);
  }
  #[test]
  fn expr() {
    assert_case(r#"DirectedEdge[x, y, z]; a \[DirectedEdge] b"#, r#"a  b"#);
  }
  #[test]
  fn undirected_edge() {
    assert_case(r#"UndirectedEdge[x, y, z]"#, r#"UndirectedEdge[x, y, z]"#);
  }
  #[test]
  fn less() {
    assert_case(r#"UndirectedEdge[x, y, z]; a <-> b"#, r#"a <-> b"#);
  }
}

mod vertex_in_degree {
  use super::*;

  #[test]
  fn directed_edge_list() {
    assert_eq!(
      interpret("VertexInDegree[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}]").unwrap(),
      "{1, 1, 2}"
    );
  }

  #[test]
  fn vertex_order_follows_first_appearance() {
    assert_eq!(
      interpret("VertexInDegree[{3 -> 2, 1 -> 3, 2 -> 1}]").unwrap(),
      "{1, 1, 1}"
    );
  }

  #[test]
  fn graph_with_explicit_vertices() {
    assert_eq!(
      interpret(
        "VertexInDegree[Graph[{a, b, c, d}, \
         {a -> b, b -> c, c -> a, a -> c, d -> a}]]"
      )
      .unwrap(),
      "{2, 1, 2, 0}"
    );
  }

  #[test]
  fn directed_self_loop_counts_once() {
    assert_eq!(
      interpret("VertexInDegree[{1 -> 1, 1 -> 2, 2 -> 1}]").unwrap(),
      "{2, 1}"
    );
  }

  #[test]
  fn pure_undirected_counts_both_endpoints() {
    assert_eq!(
      interpret("VertexInDegree[{1 <-> 2, 2 <-> 3, 3 <-> 1}]").unwrap(),
      "{2, 2, 2}"
    );
  }

  #[test]
  fn undirected_self_loop_counts_twice() {
    assert_eq!(
      interpret("VertexInDegree[{1 <-> 1, 1 <-> 2}]").unwrap(),
      "{3, 1}"
    );
  }

  #[test]
  fn mixed_graph_ignores_undirected_edges() {
    // A graph with any directed edge is a mixed graph: undirected edges
    // contribute nothing to in-degree.
    assert_eq!(
      interpret("VertexInDegree[{1 -> 2, 2 <-> 3, 3 -> 1}]").unwrap(),
      "{1, 1, 0}"
    );
    assert_eq!(
      interpret("VertexInDegree[{1 <-> 2, 2 <-> 3, 3 <-> 1, 1 -> 2}]").unwrap(),
      "{0, 1, 0}"
    );
  }

  #[test]
  fn directed_and_undirected_edge_heads() {
    assert_eq!(
      interpret(
        "VertexInDegree[{DirectedEdge[a, b], \
         UndirectedEdge[b, c], DirectedEdge[c, a]}]"
      )
      .unwrap(),
      "{1, 1, 0}"
    );
  }

  #[test]
  fn single_vertex_query() {
    assert_eq!(
      interpret("VertexInDegree[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}, 1]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "VertexInDegree[Graph[{a, b, c}, {a -> b, b -> c, c -> a, a -> c}], c]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn no_edges() {
    assert_eq!(
      interpret("VertexInDegree[Graph[{1, 2, 3}, {}]]").unwrap(),
      "{0, 0, 0}"
    );
  }

  #[test]
  fn unknown_vertex_stays_unevaluated() {
    assert_eq!(
      interpret("VertexInDegree[{1 -> 2, 2 -> 3}, 5]").unwrap(),
      "VertexInDegree[{1 -> 2, 2 -> 3}, 5]"
    );
  }
}

mod vertex_out_degree {
  use super::*;

  #[test]
  fn directed_edge_list() {
    assert_eq!(
      interpret("VertexOutDegree[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}]").unwrap(),
      "{2, 1, 1}"
    );
  }

  #[test]
  fn graph_with_explicit_vertices() {
    assert_eq!(
      interpret("VertexOutDegree[Graph[{1, 2, 3}, {1 -> 2, 2 -> 3}]]").unwrap(),
      "{1, 1, 0}"
    );
  }

  #[test]
  fn directed_self_loop_counts_once() {
    assert_eq!(
      interpret("VertexOutDegree[{1 -> 1, 1 -> 2, 2 -> 1}]").unwrap(),
      "{2, 1}"
    );
  }

  #[test]
  fn pure_undirected_counts_both_endpoints() {
    assert_eq!(
      interpret("VertexOutDegree[{1 <-> 2, 2 <-> 3, 3 <-> 1}]").unwrap(),
      "{2, 2, 2}"
    );
  }

  #[test]
  fn undirected_self_loop_counts_twice() {
    assert_eq!(
      interpret("VertexOutDegree[{1 <-> 1, 1 <-> 2}]").unwrap(),
      "{3, 1}"
    );
  }

  #[test]
  fn mixed_graph_ignores_undirected_edges() {
    assert_eq!(
      interpret("VertexOutDegree[{1 -> 2, 2 <-> 3, 3 -> 1}]").unwrap(),
      "{1, 0, 1}"
    );
  }

  #[test]
  fn single_vertex_query() {
    assert_eq!(
      interpret("VertexOutDegree[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}, 1]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn unknown_vertex_stays_unevaluated() {
    assert_eq!(
      interpret("VertexOutDegree[{1 -> 2, 2 -> 3}, 5]").unwrap(),
      "VertexOutDegree[{1 -> 2, 2 -> 3}, 5]"
    );
  }
}

mod weighted_adjacency_matrix {
  use super::*;

  #[test]
  fn directed_with_weights() {
    assert_eq!(
      // Normal[...] collapses wolframscript's SparseArray to the dense matrix
      // Woxi already returns, keeping the conformance comparison meaningful.
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2, 3}, {1 -> 2, 2 -> 3}, EdgeWeight -> {5, 10}]]]"
      )
      .unwrap(),
      "{{0, 5, 0}, {0, 0, 10}, {0, 0, 0}}"
    );
  }

  #[test]
  fn undirected_with_weights() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2, 3}, {1 <-> 2, 2 <-> 3, 1 <-> 3}, \
         EdgeWeight -> {5, 10, 2}]]]"
      )
      .unwrap(),
      "{{0, 5, 2}, {5, 0, 10}, {2, 10, 0}}"
    );
  }

  #[test]
  fn implicit_vertices() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1 <-> 2, 2 <-> 3}, EdgeWeight -> {5, 10}]]]"
      )
      .unwrap(),
      "{{0, 5, 0}, {5, 0, 10}, {0, 10, 0}}"
    );
  }

  #[test]
  fn default_weight_is_one() {
    assert_eq!(
      interpret("Normal[WeightedAdjacencyMatrix[Graph[{1 -> 2, 2 -> 3}]]]")
        .unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}"
    );
  }

  #[test]
  fn directed_self_loop() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2}, {1 -> 1, 1 -> 2}, EdgeWeight -> {7, 3}]]]"
      )
      .unwrap(),
      "{{7, 3}, {0, 0}}"
    );
  }

  #[test]
  fn undirected_self_loop_counts_once() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2}, {1 <-> 1, 1 <-> 2}, EdgeWeight -> {7, 3}]]]"
      )
      .unwrap(),
      "{{7, 3}, {3, 0}}"
    );
  }

  #[test]
  fn parallel_edges_sum_weights() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2, 3}, {1 -> 2, 1 -> 2, 2 -> 3}, \
         EdgeWeight -> {5, 4, 10}]]]"
      )
      .unwrap(),
      "{{0, 9, 0}, {0, 0, 10}, {0, 0, 0}}"
    );
  }

  #[test]
  fn symbolic_weight() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2}, {1 <-> 2}, EdgeWeight -> {x}]]]"
      )
      .unwrap(),
      "{{0, x}, {x, 0}}"
    );
  }

  #[test]
  fn real_weight() {
    assert_eq!(
      interpret(
        "Normal[WeightedAdjacencyMatrix[\
         Graph[{1, 2}, {1 <-> 2}, EdgeWeight -> {2.5}]]]"
      )
      .unwrap(),
      "{{0, 2.5}, {2.5, 0}}"
    );
  }
}

mod find_shortest_path {
  use super::*;

  #[test]
  fn weighted_directed_dijkstra() {
    // Shortest weighted path a -> e in the RosettaCode Dijkstra example.
    assert_eq!(
      interpret(
        "FindShortestPath[Graph[{\"a\" \\[DirectedEdge] \"b\", \
         \"a\" \\[DirectedEdge] \"c\", \"b\" \\[DirectedEdge] \"c\", \
         \"b\" \\[DirectedEdge] \"d\", \"c\" \\[DirectedEdge] \"d\", \
         \"d\" \\[DirectedEdge] \"e\", \"a\" \\[DirectedEdge] \"f\", \
         \"c\" \\[DirectedEdge] \"f\", \"e\" \\[DirectedEdge] \"f\"}, \
         EdgeWeight -> {7, 9, 10, 15, 11, 6, 14, 2, 9}], \"a\", \"e\"]"
      )
      .unwrap(),
      "{a, c, d, e}"
    );
  }

  #[test]
  fn unreachable_returns_empty() {
    assert_eq!(
      interpret(
        "FindShortestPath[Graph[{\"a\" \\[DirectedEdge] \"b\"}], \"b\", \"a\"]"
      )
      .unwrap(),
      "{}"
    );
  }
}

mod transitive_closure_graph {
  use super::*;

  #[test]
  fn directed_closures() {
    assert_eq!(
      interpret("TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3}]]").unwrap(),
      "Graph[<3>, <3>]"
    );
    assert_eq!(
      interpret("EdgeList[TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3}]]]")
        .unwrap(),
      "{1 \u{f3d5} 2, 1 \u{f3d5} 3, 2 \u{f3d5} 3}"
    );
    // A cycle closes into the complete digraph without self-loops
    assert_eq!(
      interpret(
        "EdgeList[TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]]"
      )
      .unwrap(),
      "{1 \u{f3d5} 2, 1 \u{f3d5} 3, 2 \u{f3d5} 1, 2 \u{f3d5} 3, 3 \u{f3d5} 1, 3 \u{f3d5} 2}"
    );
    // Components stay separate; diamond DAG closes with the long edge
    assert_eq!(
      interpret("EdgeList[TransitiveClosureGraph[Graph[{1 -> 2, 3 -> 4}]]]")
        .unwrap(),
      "{1 \u{f3d5} 2, 3 \u{f3d5} 4}"
    );
    assert_eq!(
      interpret("EdgeList[TransitiveClosureGraph[Graph[{a -> b, b -> c}]]]")
        .unwrap(),
      "{a \u{f3d5} b, a \u{f3d5} c, b \u{f3d5} c}"
    );
  }

  #[test]
  fn undirected_closures_connect_components() {
    assert_eq!(
      interpret("EdgeList[TransitiveClosureGraph[Graph[{1 <-> 2, 2 <-> 3}]]]")
        .unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 2 \u{f3d4} 3}"
    );
  }

  #[test]
  fn edge_lists_and_invalid_input() {
    // Raw edge lists wrap into a Graph first
    assert_eq!(
      interpret("TransitiveClosureGraph[{1 -> 2}]").unwrap(),
      "Graph[<2>, <1>]"
    );
    assert_eq!(
      interpret("TransitiveClosureGraph[x]").unwrap(),
      "TransitiveClosureGraph[x]"
    );
  }
}

mod find_independent_vertex_set {
  use super::*;

  #[test]
  fn maximum_sets() {
    assert_eq!(
      interpret(
        "FindIndependentVertexSet[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 <-> 1}]]"
      )
      .unwrap(),
      "{{1, 3}}"
    );
    assert_eq!(
      interpret("FindIndependentVertexSet[Graph[{1 <-> 2, 1 <-> 3, 1 <-> 4}]]")
        .unwrap(),
      "{{2, 3, 4}}"
    );
    assert_eq!(
      interpret(
        "FindIndependentVertexSet[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 <-> 5}]]"
      )
      .unwrap(),
      "{{1, 3, 5}}"
    );
    assert_eq!(
      interpret(
        "FindIndependentVertexSet[Graph[{1 <-> 2, 1 <-> 3, 2 <-> 3, 4 <-> 5}]]"
      )
      .unwrap(),
      "{{1, 4}}"
    );
    assert_eq!(
      interpret("FindIndependentVertexSet[Graph[{a <-> b, b <-> c}]]").unwrap(),
      "{{a, c}}"
    );
  }

  #[test]
  fn tie_break_uses_vertex_list_order() {
    // Vertex 2 appears first in the vertex list, so it wins the tie
    assert_eq!(
      interpret("FindIndependentVertexSet[Graph[{2 <-> 1}]]").unwrap(),
      "{{2}}"
    );
    assert_eq!(
      interpret(
        "FindIndependentVertexSet[Graph[{5 <-> 3, 3 <-> 1, 1 <-> 4, 4 <-> 2}]]"
      )
      .unwrap(),
      "{{5, 1, 2}}"
    );
  }

  #[test]
  fn directed_and_invalid_input() {
    // Directed graphs use the underlying undirected graph
    assert_eq!(
      interpret("FindIndependentVertexSet[Graph[{1 -> 2, 2 -> 3}]]").unwrap(),
      "{{1, 3}}"
    );
    assert_eq!(
      interpret("FindIndependentVertexSet[x]").unwrap(),
      "FindIndependentVertexSet[x]"
    );
  }
}

mod vertex_component {
  use super::*;

  #[test]
  fn undirected_components() {
    assert_eq!(
      interpret("VertexComponent[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}], 1]")
        .unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}], {4}]")
        .unwrap(),
      "{4, 5}"
    );
    // BFS order from the seed, not vertex-list order
    assert_eq!(
      interpret("VertexComponent[Graph[{5 <-> 3, 3 <-> 1, 8 <-> 9}], 1]")
        .unwrap(),
      "{1, 3, 5}"
    );
    assert_eq!(
      interpret(
        "VertexComponent[Graph[{9 <-> 7, 7 <-> 5, 5 <-> 3, 2 <-> 4}], 5]"
      )
      .unwrap(),
      "{5, 7, 3, 9}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{a <-> b, c <-> d}], b]").unwrap(),
      "{b, a}"
    );
  }

  #[test]
  fn directed_in_components() {
    // Directed graphs return the vertices that can REACH the seed
    assert_eq!(
      interpret("VertexComponent[Graph[{1 -> 2, 3 -> 2}], 1]").unwrap(),
      "{1}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{1 -> 2, 3 -> 2}], 2]").unwrap(),
      "{2, 1, 3}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{1 -> 2, 2 -> 1, 2 -> 3}], 1]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{1 -> 2, 2 -> 3, 4 -> 3, 5 -> 4}], 3]")
        .unwrap(),
      "{3, 2, 4, 1, 5}"
    );
  }

  #[test]
  fn multiple_seeds_and_invalid_vertices() {
    // Seeds expand sequentially with shared visited state
    assert_eq!(
      interpret("VertexComponent[Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}], {1, 4}]")
        .unwrap(),
      "{1, 2, 3, 4, 5}"
    );
    assert_eq!(
      interpret("VertexComponent[Graph[{1 -> 2}], {2, 1}]").unwrap(),
      "{2, 1}"
    );
    // Unknown vertex emits VertexComponent::inv and stays unevaluated
    assert_eq!(
      interpret("VertexComponent[Graph[{1 <-> 2}], 7]").unwrap(),
      "VertexComponent[Graph[<2>, <1>], 7]"
    );
  }
}

mod weighted_adjacency_graph {
  use super::*;

  #[test]
  fn symmetric_matrices_give_undirected_graphs() {
    assert_eq!(
      interpret("WeightedAdjacencyGraph[{{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}]").unwrap(),
      "Graph[<3>, <2>]"
    );
    assert_eq!(
      interpret("EdgeList[WeightedAdjacencyGraph[{{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}]]").unwrap(),
      "{1 \u{f3d4} 2, 2 \u{f3d4} 3}"
    );
    // Custom vertex names
    assert_eq!(
      interpret("VertexList[WeightedAdjacencyGraph[{a, b, c}, {{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}]]").unwrap(),
      "{a, b, c}"
    );
    // Diagonal entries are self-loops; zero is a real weight, only
    // Infinity marks absence
    assert_eq!(
      interpret("EdgeList[WeightedAdjacencyGraph[{{1, 2}, {2, Infinity}}]]")
        .unwrap(),
      "{1 \u{f3d4} 1, 1 \u{f3d4} 2}"
    );
  }

  #[test]
  fn asymmetric_matrices_give_directed_graphs() {
    assert_eq!(
      interpret(
        "EdgeList[WeightedAdjacencyGraph[{{Infinity, 2}, {3, Infinity}}]]"
      )
      .unwrap(),
      "{1 \u{f3d5} 2, 2 \u{f3d5} 1}"
    );
  }

  #[test]
  fn weights_survive_round_trips() {
    assert_eq!(
      interpret("Normal[WeightedAdjacencyMatrix[WeightedAdjacencyGraph[{{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}]]]").unwrap(),
      "{{0, 2, 0}, {2, 0, 5}, {0, 5, 0}}"
    );
  }

  #[test]
  fn invalid_input() {
    assert_eq!(
      interpret("WeightedAdjacencyGraph[x]").unwrap(),
      "WeightedAdjacencyGraph[x]"
    );
  }
}

mod graph_distance_weighted {
  use super::*;

  #[test]
  fn uses_edge_weights_and_returns_reals() {
    // Regression: EdgeWeight was ignored (returned hop count 2)
    assert_eq!(
      interpret("GraphDistance[WeightedAdjacencyGraph[{{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}], 1, 3]").unwrap(),
      "7."
    );
    assert_eq!(
      interpret("GraphDistance[WeightedAdjacencyGraph[{{Infinity, 2}, {3, Infinity}}], 2, 1]").unwrap(),
      "3."
    );
    // Dijkstra takes the cheap two-hop path over the expensive direct
    // edge
    assert_eq!(
      interpret("GraphDistance[WeightedAdjacencyGraph[{{Infinity, 1, 10}, {1, Infinity, 1}, {10, 1, Infinity}}], 1, 3]").unwrap(),
      "2."
    );
    // Unreachable stays Infinity; unweighted graphs keep integer hops
    assert_eq!(
      interpret("GraphDistance[WeightedAdjacencyGraph[{{Infinity, 2}, {Infinity, Infinity}}], 2, 1]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("GraphDistance[Graph[{1 <-> 2, 2 <-> 3}], 1, 3]").unwrap(),
      "2"
    );
  }
}

mod find_minimum_cost_flow {
  use super::*;

  #[test]
  fn unit_capacity_min_cost_max_flow() {
    // Flow 2: the direct edge (4) plus the two-hop path (2 + 1)
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 2, 4}, {0, 0, 1}, {0, 0, 0}}, 1, 3]")
        .unwrap(),
      "7"
    );
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 1}, {0, 0}}, 1, 2]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 5, 2, 0}, {0, 0, 0, 3}, {0, 0, 0, 4}, {0, 0, 0, 0}}, 1, 4]").unwrap(),
      "14"
    );
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 1, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}}, 1, 4]").unwrap(),
      "4"
    );
    // Reverse-direction entries are separate arcs
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 3}, {2, 0}}, 1, 2]").unwrap(),
      "3"
    );
    // Residual rerouting: diamond with a cross edge
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 1, 2, 0}, {0, 0, 1, 2}, {0, 0, 0, 1}, {0, 0, 0, 0}}, 1, 4]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 9, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 9, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 0, 0}}, 1, 5]").unwrap(),
      "13"
    );
  }

  #[test]
  fn real_costs_give_reals() {
    assert_eq!(
      interpret(
        "FindMinimumCostFlow[{{0, 1.5, 0}, {0, 0, 2.5}, {0, 0, 0}}, 1, 3]"
      )
      .unwrap(),
      "4."
    );
  }

  #[test]
  fn no_flow_stays_unevaluated() {
    assert_eq!(
      interpret("FindMinimumCostFlow[{{0, 0}, {0, 0}}, 1, 2]").unwrap(),
      "FindMinimumCostFlow[{{0, 0}, {0, 0}}, 1, 2]"
    );
    assert_eq!(
      interpret("FindMinimumCostFlow[x, 1, 2]").unwrap(),
      "FindMinimumCostFlow[x, 1, 2]"
    );
  }
}

mod nearest_neighbor_graph {
  use super::*;

  #[test]
  fn nearest_neighbor_edges() {
    assert_eq!(
      interpret("NearestNeighborGraph[{{0, 0}, {1, 0}, {5, 5}, {6, 5}}]")
        .unwrap(),
      "Graph[<4>, <2>]"
    );
    assert_eq!(
      interpret(
        "EdgeList[NearestNeighborGraph[{{0, 0}, {1, 0}, {5, 5}, {6, 5}}]]"
      )
      .unwrap(),
      "{{0, 0} \u{f3d4} {1, 0}, {5, 5} \u{f3d4} {6, 5}}"
    );
    // Mutual nearest neighbors collapse into single undirected edges
    assert_eq!(
      interpret("EdgeList[NearestNeighborGraph[{1, 2, 4, 8}]]").unwrap(),
      "{1 \u{f3d4} 2, 2 \u{f3d4} 4, 4 \u{f3d4} 8}"
    );
    // Vertices keep input order even when scrambled
    assert_eq!(
      interpret("EdgeList[NearestNeighborGraph[{10, 3, 7}]]").unwrap(),
      "{10 \u{f3d4} 7, 3 \u{f3d4} 7}"
    );
    assert_eq!(
      interpret("VertexList[NearestNeighborGraph[{1, 2, 4, 8}]]").unwrap(),
      "{1, 2, 4, 8}"
    );
  }

  #[test]
  fn ties_include_all_equidistant_points() {
    assert_eq!(
      interpret("EdgeList[NearestNeighborGraph[{0, 1, 2, 3}]]").unwrap(),
      "{0 \u{f3d4} 1, 1 \u{f3d4} 2, 2 \u{f3d4} 3}"
    );
    assert_eq!(
      interpret("EdgeList[NearestNeighborGraph[{0, 2, 4}]]").unwrap(),
      "{0 \u{f3d4} 2, 2 \u{f3d4} 4}"
    );
  }

  #[test]
  fn k_nearest_and_real_coordinates() {
    assert_eq!(
      interpret("EdgeList[NearestNeighborGraph[{1, 2, 4, 8}, 2]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 4, 2 \u{f3d4} 4, 2 \u{f3d4} 8, 4 \u{f3d4} 8}"
    );
    assert_eq!(
      interpret(
        "EdgeList[NearestNeighborGraph[{{0, 0}, {2, 0}, {4, 1}, {1, 3}}, 2]]"
      )
      .unwrap(),
      "{{0, 0} \u{f3d4} {2, 0}, {0, 0} \u{f3d4} {1, 3}, {2, 0} \u{f3d4} {4, 1}, {2, 0} \u{f3d4} {1, 3}, {4, 1} \u{f3d4} {1, 3}}"
    );
    assert_eq!(
      interpret(
        "EdgeList[NearestNeighborGraph[{{0., 0.}, {1.5, 0.}, {1.5, 1.}}]]"
      )
      .unwrap(),
      "{{0., 0.} \u{f3d4} {1.5, 0.}, {1.5, 0.} \u{f3d4} {1.5, 1.}}"
    );
  }

  #[test]
  fn invalid_input() {
    // NearestNeighborGraph::list message
    assert_eq!(
      interpret("NearestNeighborGraph[x]").unwrap(),
      "NearestNeighborGraph[x]"
    );
  }
}

mod harary_graph {
  use super::*;

  #[test]
  fn displays_as_graph_summary() {
    assert_eq!(interpret("HararyGraph[2, 8]").unwrap(), "Graph[<8>, <8>]");
    assert_eq!(interpret("HararyGraph[4, 9]").unwrap(), "Graph[<9>, <18>]");
  }

  #[test]
  fn even_k_is_circulant() {
    // H_{2,n} is the cycle; H_{2r,n} connects each vertex to its r
    // nearest neighbors on each side
    assert_eq!(
      interpret("EdgeList[HararyGraph[2, 8]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 8, 2 \u{f3d4} 3, 3 \u{f3d4} 4, 4 \u{f3d4} 5, 5 \u{f3d4} 6, 6 \u{f3d4} 7, 7 \u{f3d4} 8}"
    );
    assert_eq!(interpret("EdgeCount[HararyGraph[4, 8]]").unwrap(), "16");
  }

  #[test]
  fn odd_k_even_n_adds_diameters() {
    // H_{3,8}: cycle plus the four diameters
    assert_eq!(
      interpret("EdgeList[HararyGraph[3, 8]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 5, 1 \u{f3d4} 8, 2 \u{f3d4} 3, 2 \u{f3d4} 6, 3 \u{f3d4} 4, 3 \u{f3d4} 7, 4 \u{f3d4} 5, 4 \u{f3d4} 8, 5 \u{f3d4} 6, 6 \u{f3d4} 7, 7 \u{f3d4} 8}"
    );
  }

  #[test]
  fn odd_k_odd_n_adds_half_diagonals() {
    // H_{3,7}: cycle, the (1, 1+(n-1)/2) edge, and (i, i+(n+1)/2)
    assert_eq!(
      interpret("EdgeList[HararyGraph[3, 7]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 4, 1 \u{f3d4} 5, 1 \u{f3d4} 7, 2 \u{f3d4} 3, 2 \u{f3d4} 6, 3 \u{f3d4} 4, 3 \u{f3d4} 7, 4 \u{f3d4} 5, 5 \u{f3d4} 6, 6 \u{f3d4} 7}"
    );
  }

  #[test]
  fn k_equals_n_minus_one_is_complete() {
    assert_eq!(interpret("EdgeCount[HararyGraph[7, 8]]").unwrap(), "28");
    assert_eq!(
      interpret("EdgeList[HararyGraph[2, 3]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 2 \u{f3d4} 3}"
    );
  }

  #[test]
  fn edge_count_is_ceil_kn_over_2() {
    assert_eq!(interpret("EdgeCount[HararyGraph[3, 11]]").unwrap(), "17");
    assert_eq!(interpret("VertexCount[HararyGraph[4, 9]]").unwrap(), "9");
    assert_eq!(
      interpret("VertexList[HararyGraph[3, 11]]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}"
    );
  }

  #[test]
  fn options_are_accepted_and_ignored() {
    assert_eq!(
      interpret("HararyGraph[3, 7, PlotLabel -> x]").unwrap(),
      "Graph[<7>, <11>]"
    );
  }

  #[test]
  fn k_must_be_at_least_two() {
    assert_eq!(interpret("HararyGraph[1, 5]").unwrap(), "HararyGraph[1, 5]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "HararyGraph::intg: Integer greater than 1 expected at position 1 in HararyGraph[1, 5]."
      )),
      "expected intg message, got {:?}",
      msgs
    );
  }

  #[test]
  fn n_must_exceed_k() {
    assert_eq!(interpret("HararyGraph[8, 8]").unwrap(), "HararyGraph[8, 8]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "HararyGraph::intg: Integer greater than 8 expected at position 2 in HararyGraph[8, 8]."
      )),
      "expected intg message, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_positive_and_real_arguments_warn_intpm() {
    assert_eq!(interpret("HararyGraph[0, 5]").unwrap(), "HararyGraph[0, 5]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "HararyGraph::intpm: Positive machine-sized integer expected at position 1 in HararyGraph[0, 5]."
      )),
      "expected intpm message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("HararyGraph[3, 7.5]").unwrap(),
      "HararyGraph[3, 7.5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "HararyGraph::intpm: Positive machine-sized integer expected at position 2 in HararyGraph[3, 7.5]."
      )),
      "expected intpm message, got {:?}",
      msgs
    );
  }

  #[test]
  fn symbolic_arguments_stay_unevaluated_silently() {
    assert_eq!(interpret("HararyGraph[2, n]").unwrap(), "HararyGraph[2, n]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }

  #[test]
  fn non_option_extra_argument_warns_nonopt() {
    assert_eq!(
      interpret("HararyGraph[3, 7, 1]").unwrap(),
      "HararyGraph[3, 7, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "HararyGraph::nonopt: Options expected (instead of 1) beyond position 2 in HararyGraph[3, 7, 1]."
      )),
      "expected nonopt message, got {:?}",
      msgs
    );
  }
}

mod connectivity {
  use super::*;

  #[test]
  fn edge_connectivity_basic() {
    assert_eq!(
      interpret("EdgeConnectivity[CompleteGraph[5]]").unwrap(),
      "4"
    );
    assert_eq!(interpret("EdgeConnectivity[CycleGraph[6]]").unwrap(), "2");
    assert_eq!(interpret("EdgeConnectivity[StarGraph[6]]").unwrap(), "1");
    assert_eq!(
      interpret("EdgeConnectivity[PathGraph[Range[5]]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("EdgeConnectivity[PetersenGraph[5, 2]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn vertex_connectivity_basic() {
    assert_eq!(
      interpret("VertexConnectivity[CompleteGraph[5]]").unwrap(),
      "4"
    );
    assert_eq!(interpret("VertexConnectivity[CycleGraph[6]]").unwrap(), "2");
    assert_eq!(interpret("VertexConnectivity[StarGraph[6]]").unwrap(), "1");
    assert_eq!(
      interpret("VertexConnectivity[PetersenGraph[5, 2]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("VertexConnectivity[GridGraph[{3, 3}]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn harary_graphs_are_exactly_k_connected() {
    // H_{k,n} is the minimal k-connected graph — cross-validates both
    // HararyGraph and the connectivity algorithms
    assert_eq!(
      interpret(
        "Table[{VertexConnectivity[HararyGraph[k, 9]], EdgeConnectivity[HararyGraph[k, 9]]}, {k, 2, 8}]"
      )
      .unwrap(),
      "{{2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}}"
    );
  }

  #[test]
  fn disconnected_graphs_have_zero_connectivity() {
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4}, {1 <-> 2, 3 <-> 4}]; {EdgeConnectivity[g], VertexConnectivity[g]}"
      )
      .unwrap(),
      "{0, 0}"
    );
    assert_eq!(
      interpret("VertexConnectivity[Graph[{1, 2, 3}, {}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn s_t_connectivity() {
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5, 6}, {1 <-> 2, 2 <-> 3, 3 <-> 4, 1 <-> 5, 5 <-> 6, 6 <-> 4, 2 <-> 5, 3 <-> 6}]; {EdgeConnectivity[g, 2, 6], VertexConnectivity[g, 2, 6]}"
      )
      .unwrap(),
      "{2, 2}"
    );
    // Non-adjacent pair on a cycle
    assert_eq!(
      interpret("VertexConnectivity[CycleGraph[5], 1, 3]").unwrap(),
      "2"
    );
  }

  #[test]
  fn adjacent_vertices_have_vertex_connectivity_zero() {
    // wolframscript convention: no vertex cut separates adjacent vertices
    assert_eq!(
      interpret("VertexConnectivity[CycleGraph[5], 1, 2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("VertexConnectivity[CompleteGraph[5], 1, 2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn same_vertex_artifacts() {
    // wolframscript: s == t gives the degree (edge) or edge count (vertex)
    assert_eq!(
      interpret("EdgeConnectivity[StarGraph[6], 1, 1]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("EdgeConnectivity[StarGraph[6], 3, 3]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("VertexConnectivity[CompleteGraph[5], 1, 1]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret("VertexConnectivity[CycleGraph[5], 1, 1]").unwrap(),
      "5"
    );
  }

  #[test]
  fn single_vertex_graph() {
    // wolframscript: EdgeConnectivity stays unevaluated, VertexConnectivity
    // gives 0
    assert_eq!(
      interpret("EdgeConnectivity[Graph[{1}, {}]]").unwrap(),
      "EdgeConnectivity[Graph[<1>, <0>]]"
    );
    assert_eq!(
      interpret("VertexConnectivity[Graph[{1}, {}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn invalid_vertex_emits_inv_message() {
    assert_eq!(
      interpret("EdgeConnectivity[CycleGraph[5], 1, 7]").unwrap(),
      "EdgeConnectivity[Graph[<5>, <5>], 1, 7]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "EdgeConnectivity::inv: The argument 3 in EdgeConnectivity[Graph[<5>, <5>], 1, 7] is not a valid vertex."
      )),
      "expected inv message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("VertexConnectivity[CycleGraph[5], 9, 2]").unwrap(),
      "VertexConnectivity[Graph[<5>, <5>], 9, 2]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "VertexConnectivity::inv: The argument 2 in VertexConnectivity[Graph[<5>, <5>], 9, 2] is not a valid vertex."
      )),
      "expected inv message, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_graph_argument_stays_unevaluated() {
    assert_eq!(
      interpret("EdgeConnectivity[x]").unwrap(),
      "EdgeConnectivity[x]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }
}

mod k_core_components {
  use super::*;

  #[test]
  fn finds_k_core_components() {
    // Two 3-cores joined by a path through vertex 5
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1 <-> 2, 1 <-> 3, 1 <-> 4, 2 <-> 3, 2 <-> 4, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 7, 6 <-> 8, 6 <-> 9, 7 <-> 8, 7 <-> 9, 8 <-> 9}]; KCoreComponents[g, 3]"
      )
      .unwrap(),
      "{{6, 7, 8, 9}, {1, 2, 3, 4}}"
    );
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 2]").unwrap(),
      "{{1, 2, 3, 4, 5}}"
    );
    assert_eq!(
      interpret("KCoreComponents[PetersenGraph[5, 2], 3]").unwrap(),
      "{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}"
    );
  }

  #[test]
  fn component_ordering() {
    // Size descending, ties broken by descending position of the first
    // member in the vertex list
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 4 <-> 5, 4 <-> 6, 4 <-> 7, 5 <-> 6, 5 <-> 7, 6 <-> 7, 9 <-> 10, 9 <-> 11, 10 <-> 11}]; KCoreComponents[g, 2]"
      )
      .unwrap(),
      "{{4, 5, 6, 7}, {9, 10, 11}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 10}, {1 <-> 10, 2 <-> 3}]; KCoreComponents[g, 1]"
      )
      .unwrap(),
      "{{2, 3}, {1, 10}}"
    );
    // Members appear in VertexList order, not sorted by value
    assert_eq!(
      interpret(
        "g = Graph[{5, 4, 3, 2, 1}, {5 <-> 4, 2 <-> 1}]; KCoreComponents[g, 1]"
      )
      .unwrap(),
      "{{2, 1}, {5, 4}}"
    );
  }

  #[test]
  fn k_larger_than_max_degree_gives_empty() {
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 7]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn degree_one_core_keeps_leaves() {
    assert_eq!(
      interpret("KCoreComponents[StarGraph[6], 1]").unwrap(),
      "{{1, 2, 3, 4, 5, 6}}"
    );
    assert_eq!(
      interpret("KCoreComponents[CompleteGraph[6], 5]").unwrap(),
      "{{1, 2, 3, 4, 5, 6}}"
    );
    assert_eq!(
      interpret("KCoreComponents[WheelGraph[7], 3]").unwrap(),
      "{{1, 2, 3, 4, 5, 6, 7}}"
    );
  }

  #[test]
  fn in_out_parameter_accepted() {
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 2, \"In\"]").unwrap(),
      "{{1, 2, 3, 4, 5}}"
    );
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 2, \"Out\"]").unwrap(),
      "{{1, 2, 3, 4, 5}}"
    );
  }

  #[test]
  fn invalid_parameter_emits_inv() {
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 2, \"Bogus\"]").unwrap(),
      "KCoreComponents[Graph[<5>, <5>], 2, Bogus]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KCoreComponents::inv: The argument Bogus in KCoreComponents[Graph[<5>, <5>], 2, Bogus] is not a valid parameter."
      )),
      "expected inv message, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_integer_k_emits_int() {
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5], 1.5]").unwrap(),
      "KCoreComponents[Graph[<5>, <5>], 1.5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KCoreComponents::int: Integer expected at position 2 in KCoreComponents[Graph[<5>, <5>], 1.5]."
      )),
      "expected int message, got {:?}",
      msgs
    );
  }

  #[test]
  fn wrong_arg_count_emits_argtu() {
    assert_eq!(
      interpret("KCoreComponents[CycleGraph[5]]").unwrap(),
      "KCoreComponents[Graph[<5>, <5>]]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "KCoreComponents::argtu: KCoreComponents called with 1 argument; 2 or 3 arguments are expected."
      )),
      "expected argtu message, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_graph_stays_unevaluated() {
    assert_eq!(
      interpret("KCoreComponents[x, 2]").unwrap(),
      "KCoreComponents[x, 2]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }
}

mod find_clique {
  use super::*;

  #[test]
  fn finds_largest_clique() {
    assert_eq!(
      interpret("FindClique[CompleteGraph[4]]").unwrap(),
      "{{1, 2, 3, 4}}"
    );
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4, 4 <-> 5}]; FindClique[g]"
      )
      .unwrap(),
      "{{1, 2, 3}}"
    );
    assert_eq!(interpret("FindClique[CycleGraph[6]]").unwrap(), "{{1, 2}}");
  }

  #[test]
  fn largest_means_largest_not_first() {
    // The lexicographically first maximal clique {1, 2} is smaller than
    // the triangle — the triangle must win
    assert_eq!(
      interpret(
        "g4 = Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 3 <-> 4, 3 <-> 5, 4 <-> 5}]; FindClique[g4]"
      )
      .unwrap(),
      "{{3, 4, 5}}"
    );
    // Ties go to the ascending-lexicographic first
    assert_eq!(
      interpret(
        "g2 = Graph[{1, 2, 3, 4, 5, 6}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 4 <-> 5, 4 <-> 6, 5 <-> 6}]; FindClique[g2]"
      )
      .unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn only_maximal_cliques_count() {
    // Every edge of a triangle is inside the triangle, so no maximal
    // clique has size <= 2
    assert_eq!(
      interpret(
        "g2 = Graph[{1, 2, 3, 4, 5, 6}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 4 <-> 5, 4 <-> 6, 5 <-> 6}]; FindClique[g2, 2, All]"
      )
      .unwrap(),
      "{}"
    );
  }

  #[test]
  fn all_results_sorted_size_then_reverse_lex() {
    assert_eq!(
      interpret("FindClique[CycleGraph[6], 2, All]").unwrap(),
      "{{5, 6}, {4, 5}, {3, 4}, {2, 3}, {1, 6}, {1, 2}}"
    );
    assert_eq!(
      interpret(
        "g3 = Graph[{1, 2, 3, 4, 5, 6, 7}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4, 6 <-> 7}]; FindClique[g3, 3, All]"
      )
      .unwrap(),
      "{{4, 5, 6}, {1, 2, 3}, {6, 7}, {3, 4}}"
    );
  }

  #[test]
  fn count_takes_ascending_enumeration_prefix() {
    // count k >= 2: first k maximal cliques in ascending order, then
    // sorted by size descending / lex descending
    assert_eq!(
      interpret(
        "g3 = Graph[{1, 2, 3, 4, 5, 6, 7}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4, 6 <-> 7}]; FindClique[g3, 3, 2]"
      )
      .unwrap(),
      "{{1, 2, 3}, {3, 4}}"
    );
    assert_eq!(
      interpret(
        "g3 = Graph[{1, 2, 3, 4, 5, 6, 7}, {1 <-> 2, 2 <-> 3, 3 <-> 1, 3 <-> 4, 4 <-> 5, 5 <-> 6, 6 <-> 4, 6 <-> 7}]; FindClique[g3, 3, 3]"
      )
      .unwrap(),
      "{{4, 5, 6}, {1, 2, 3}, {3, 4}}"
    );
  }

  #[test]
  fn size_specifications() {
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4, 4 <-> 5}]; {FindClique[g, 2], FindClique[g, {2}], FindClique[g, {1, 2}]}"
      )
      .unwrap(),
      "{{{3, 4}}, {{3, 4}}, {{3, 4}}}"
    );
    assert_eq!(
      interpret("FindClique[CompleteGraph[5], {3}, All]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("FindClique[CycleGraph[5], Infinity, All]").unwrap(),
      "{{4, 5}, {3, 4}, {2, 3}, {1, 5}, {1, 2}}"
    );
  }

  #[test]
  fn isolated_vertices_are_one_cliques() {
    assert_eq!(interpret("FindClique[Graph[{1, 2}, {}]]").unwrap(), "{{1}}");
    assert_eq!(interpret("FindClique[Graph[{}, {}]]").unwrap(), "{}");
  }

  #[test]
  fn invalid_spec_emits_inv() {
    assert_eq!(
      interpret("FindClique[CycleGraph[5], 0]").unwrap(),
      "FindClique[Graph[<5>, <5>], 0]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "FindClique::inv: The argument 0 in FindClique[Graph[<5>, <5>], 0] is not a valid parameter."
      )),
      "expected inv message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("FindClique[CycleGraph[5], {1, 2, 3}]").unwrap(),
      "FindClique[Graph[<5>, <5>], {1, 2, 3}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "FindClique::inv: The argument {1, 2, 3} in FindClique[Graph[<5>, <5>], {1, 2, 3}] is not a valid parameter."
      )),
      "expected inv message, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_graph_stays_unevaluated() {
    assert_eq!(interpret("FindClique[x]").unwrap(), "FindClique[x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }
}

mod petersen_graph_labeling {
  use super::*;

  #[test]
  fn matches_wolfram_vertex_convention() {
    // Regression: wolframscript labels the k-jump ring 1..n and the plain
    // cycle n+1..2n (Woxi previously had them swapped), with edges sorted
    assert_eq!(
      interpret("EdgeList[PetersenGraph[5, 2]]").unwrap(),
      "{1 \u{f3d4} 3, 1 \u{f3d4} 4, 1 \u{f3d4} 6, 2 \u{f3d4} 4, 2 \u{f3d4} 5, 2 \u{f3d4} 7, 3 \u{f3d4} 5, 3 \u{f3d4} 8, 4 \u{f3d4} 9, 5 \u{f3d4} 10, 6 \u{f3d4} 7, 6 \u{f3d4} 10, 7 \u{f3d4} 8, 8 \u{f3d4} 9, 9 \u{f3d4} 10}"
    );
    assert_eq!(
      interpret("FindClique[PetersenGraph[5, 2]]").unwrap(),
      "{{1, 3}}"
    );
  }
}

mod subgraph {
  use super::*;

  #[test]
  fn induced_subgraph() {
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4, 5}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4, 4 <-> 5}]; EdgeList[Subgraph[g, {1, 2, 3}]]"
      )
      .unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 2 \u{f3d4} 3}"
    );
    assert_eq!(
      interpret("EdgeList[Subgraph[CompleteGraph[5], {2, 3, 5}]]").unwrap(),
      "{2 \u{f3d4} 3, 2 \u{f3d4} 5, 3 \u{f3d4} 5}"
    );
    assert_eq!(
      interpret("EdgeList[Subgraph[PetersenGraph[5, 2], {1, 3, 5, 6}]]")
        .unwrap(),
      "{1 \u{f3d4} 3, 3 \u{f3d4} 5, 1 \u{f3d4} 6}"
    );
  }

  #[test]
  fn keeps_given_vertex_order() {
    assert_eq!(
      interpret(
        "g = CycleGraph[6]; {VertexList[Subgraph[g, {5, 2, 1}]], EdgeList[Subgraph[g, {5, 2, 1}]]}"
      )
      .unwrap(),
      "{{5, 2, 1}, {2 \u{f3d4} 1}}"
    );
    assert_eq!(
      interpret("EdgeList[Subgraph[CycleGraph[6], {4, 3, 2}]]").unwrap(),
      "{4 \u{f3d4} 3, 3 \u{f3d4} 2}"
    );
  }

  #[test]
  fn unknown_vertices_ignored() {
    assert_eq!(
      interpret("Subgraph[CycleGraph[5], {1, 9}]").unwrap(),
      "Graph[<1>, <0>]"
    );
    assert_eq!(
      interpret("Subgraph[CycleGraph[5], {}]").unwrap(),
      "Graph[<0>, <0>]"
    );
  }

  #[test]
  fn single_vertex_spec_and_non_graph() {
    assert_eq!(
      interpret("Subgraph[CycleGraph[5], 3]").unwrap(),
      "Graph[<1>, <0>]"
    );
    assert_eq!(interpret("Subgraph[x, {1}]").unwrap(), "Subgraph[x, {1}]");
  }
}

mod line_graph {
  use super::*;

  #[test]
  fn line_graph_of_generators() {
    // Vertices index the edges in EdgeList order
    assert_eq!(
      interpret(
        "{VertexList[LineGraph[CycleGraph[4]]], EdgeList[LineGraph[CycleGraph[4]]]}"
      )
      .unwrap(),
      "{{1, 2, 3, 4}, {1 \u{f3d4} 2, 1 \u{f3d4} 3, 2 \u{f3d4} 4, 3 \u{f3d4} 4}}"
    );
    assert_eq!(
      interpret("EdgeList[LineGraph[CycleGraph[5]]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 2 \u{f3d4} 5, 3 \u{f3d4} 4, 4 \u{f3d4} 5}"
    );
    // The line graph of a star is a complete graph
    assert_eq!(
      interpret("EdgeList[LineGraph[StarGraph[5]]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 1 \u{f3d4} 4, 2 \u{f3d4} 3, 2 \u{f3d4} 4, 3 \u{f3d4} 4}"
    );
  }

  #[test]
  fn line_graph_of_complete_graph() {
    assert_eq!(
      interpret("LineGraph[CompleteGraph[4]]").unwrap(),
      "Graph[<6>, <12>]"
    );
  }

  #[test]
  fn edgeless_and_non_graph() {
    assert_eq!(
      interpret("LineGraph[Graph[{1, 2}, {}]]").unwrap(),
      "Graph[<0>, <0>]"
    );
    assert_eq!(interpret("LineGraph[x]").unwrap(), "LineGraph[x]");
  }
}

mod cycle_graph_edge_order {
  use super::*;

  #[test]
  fn edges_are_sorted() {
    // Regression: the wrap-around edge previously appeared last as n <-> 1
    assert_eq!(
      interpret("EdgeList[CycleGraph[4]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 4, 2 \u{f3d4} 3, 3 \u{f3d4} 4}"
    );
    assert_eq!(
      interpret("EdgeList[CycleGraph[6]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 6, 2 \u{f3d4} 3, 3 \u{f3d4} 4, 4 \u{f3d4} 5, 5 \u{f3d4} 6}"
    );
  }

  #[test]
  fn degenerate_cycles_are_literal() {
    // wolframscript: CycleGraph[1] has a self-loop, CycleGraph[2] a
    // doubled edge
    assert_eq!(
      interpret("EdgeList[CycleGraph[1]]").unwrap(),
      "{1 \u{f3d4} 1}"
    );
    assert_eq!(
      interpret("EdgeList[CycleGraph[2]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 2}"
    );
  }
}

mod neighborhood_graph {
  use super::*;

  #[test]
  fn center_and_neighbors() {
    assert_eq!(
      interpret(
        "{VertexList[NeighborhoodGraph[CycleGraph[6], 1]], EdgeList[NeighborhoodGraph[CycleGraph[6], 1]]}"
      )
      .unwrap(),
      "{{1, 2, 6}, {1 \u{f3d4} 2, 1 \u{f3d4} 6}}"
    );
    assert_eq!(
      interpret(
        "{VertexList[NeighborhoodGraph[PetersenGraph[5, 2], 1]], EdgeList[NeighborhoodGraph[PetersenGraph[5, 2], 1]]}"
      )
      .unwrap(),
      "{{1, 3, 4, 6}, {1 \u{f3d4} 3, 1 \u{f3d4} 4, 1 \u{f3d4} 6}}"
    );
  }

  #[test]
  fn center_incident_edges_come_first() {
    // Center 3 of K4: incident edges first with the center endpoint
    // first, then the remaining edges in canonical order
    assert_eq!(
      interpret("EdgeList[NeighborhoodGraph[CompleteGraph[4], 3]]").unwrap(),
      "{3 \u{f3d4} 1, 3 \u{f3d4} 2, 3 \u{f3d4} 4, 1 \u{f3d4} 2, 1 \u{f3d4} 4, 2 \u{f3d4} 4}"
    );
    assert_eq!(
      interpret("EdgeList[NeighborhoodGraph[WheelGraph[6], 1]]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 3, 1 \u{f3d4} 4, 1 \u{f3d4} 5, 1 \u{f3d4} 6, 2 \u{f3d4} 3, 2 \u{f3d4} 6, 3 \u{f3d4} 4, 4 \u{f3d4} 5, 5 \u{f3d4} 6}"
    );
  }

  #[test]
  fn multiple_centers() {
    assert_eq!(
      interpret(
        "g = CycleGraph[8]; {VertexList[NeighborhoodGraph[g, {1, 4}]], EdgeList[NeighborhoodGraph[g, {1, 4}]]}"
      )
      .unwrap(),
      "{{1, 4, 2, 8, 3, 5}, {1 \u{f3d4} 2, 1 \u{f3d4} 8, 4 \u{f3d4} 3, 4 \u{f3d4} 5, 2 \u{f3d4} 3}}"
    );
  }

  #[test]
  fn radius_parameter() {
    assert_eq!(
      interpret(
        "{VertexList[NeighborhoodGraph[CycleGraph[8], 1, 2]], EdgeList[NeighborhoodGraph[CycleGraph[8], 1, 2]]}"
      )
      .unwrap(),
      "{{1, 2, 3, 7, 8}, {1 \u{f3d4} 2, 1 \u{f3d4} 8, 2 \u{f3d4} 3, 7 \u{f3d4} 8}}"
    );
    // Radius 0 keeps only the center
    assert_eq!(
      interpret("VertexList[NeighborhoodGraph[CycleGraph[5], 2, 0]]").unwrap(),
      "{2}"
    );
  }

  #[test]
  fn unknown_center_and_non_graph() {
    assert_eq!(
      interpret("NeighborhoodGraph[CycleGraph[5], 9]").unwrap(),
      "Graph[<0>, <0>]"
    );
    assert_eq!(
      interpret("NeighborhoodGraph[x, 1]").unwrap(),
      "NeighborhoodGraph[x, 1]"
    );
  }
}

mod graph_predicates {
  use super::*;

  #[test]
  fn hamiltonian_graph_q() {
    assert_eq!(
      interpret("HamiltonianGraphQ[CycleGraph[5]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[CompleteGraph[4]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[PathGraph[Range[4]]]").unwrap(),
      "False"
    );
    // The Petersen graph is famously non-Hamiltonian
    assert_eq!(
      interpret("HamiltonianGraphQ[PetersenGraph[5, 2]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[StarGraph[4]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[WheelGraph[7]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[GridGraph[{3, 3}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn hamiltonian_degenerate_cases() {
    // wolframscript: K1 and the doubled-edge 2-cycle are Hamiltonian,
    // the null graph and a simple K2 are not
    assert_eq!(
      interpret("HamiltonianGraphQ[Graph[{1}, {}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[CycleGraph[1]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[CycleGraph[2]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[Graph[{}, {}]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("HamiltonianGraphQ[Graph[{1, 2}, {1 <-> 2}]]").unwrap(),
      "False"
    );
    assert_eq!(interpret("HamiltonianGraphQ[x]").unwrap(), "False");
  }

  #[test]
  fn bipartite_graph_q() {
    assert_eq!(interpret("BipartiteGraphQ[CycleGraph[4]]").unwrap(), "True");
    assert_eq!(
      interpret("BipartiteGraphQ[CycleGraph[5]]").unwrap(),
      "False"
    );
    assert_eq!(interpret("BipartiteGraphQ[StarGraph[5]]").unwrap(), "True");
    assert_eq!(
      interpret("BipartiteGraphQ[PetersenGraph[5, 2]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("BipartiteGraphQ[GridGraph[{3, 3}]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("BipartiteGraphQ[Graph[{}, {}]]").unwrap(), "True");
    assert_eq!(interpret("BipartiteGraphQ[x]").unwrap(), "False");
  }

  #[test]
  fn complete_graph_q() {
    assert_eq!(
      interpret("CompleteGraphQ[CompleteGraph[4]]").unwrap(),
      "True"
    );
    // K3 is a triangle
    assert_eq!(interpret("CompleteGraphQ[CycleGraph[3]]").unwrap(), "True");
    assert_eq!(interpret("CompleteGraphQ[CycleGraph[4]]").unwrap(), "False");
    assert_eq!(interpret("CompleteGraphQ[Graph[{1}, {}]]").unwrap(), "True");
    assert_eq!(interpret("CompleteGraphQ[Graph[{}, {}]]").unwrap(), "True");
    // A doubled edge disqualifies completeness
    assert_eq!(
      interpret("CompleteGraphQ[Graph[{1, 2}, {1 <-> 2, 1 <-> 2}]]").unwrap(),
      "False"
    );
    assert_eq!(interpret("CompleteGraphQ[x]").unwrap(), "False");
  }

  #[test]
  fn loop_free_and_simple_q() {
    assert_eq!(interpret("LoopFreeGraphQ[CycleGraph[5]]").unwrap(), "True");
    assert_eq!(
      interpret("LoopFreeGraphQ[Graph[{1, 2}, {1 <-> 1, 1 <-> 2}]]").unwrap(),
      "False"
    );
    // Multi-edges are loop-free but not simple
    assert_eq!(interpret("LoopFreeGraphQ[CycleGraph[2]]").unwrap(), "True");
    assert_eq!(interpret("SimpleGraphQ[CycleGraph[2]]").unwrap(), "False");
    assert_eq!(interpret("SimpleGraphQ[CycleGraph[1]]").unwrap(), "False");
    assert_eq!(interpret("SimpleGraphQ[CycleGraph[5]]").unwrap(), "True");
    assert_eq!(
      interpret("SimpleGraphQ[Graph[{1, 2}, {1 <-> 2, 1 <-> 2}]]").unwrap(),
      "False"
    );
    assert_eq!(interpret("LoopFreeGraphQ[x]").unwrap(), "False");
    assert_eq!(interpret("SimpleGraphQ[x]").unwrap(), "False");
  }

  #[test]
  fn path_graph_q() {
    assert_eq!(
      interpret("PathGraphQ[PathGraph[Range[4]]]").unwrap(),
      "True"
    );
    // wolframscript counts cycles as path graphs (all degrees <= 2)
    assert_eq!(interpret("PathGraphQ[CycleGraph[4]]").unwrap(), "True");
    assert_eq!(interpret("PathGraphQ[Graph[{1}, {}]]").unwrap(), "True");
    assert_eq!(
      interpret("PathGraphQ[Graph[{1, 2, 3}, {1 <-> 2}]]").unwrap(),
      "False"
    );
    assert_eq!(interpret("PathGraphQ[StarGraph[4]]").unwrap(), "False");
    assert_eq!(interpret("PathGraphQ[Graph[{}, {}]]").unwrap(), "False");
    // Multi-edges disqualify
    assert_eq!(interpret("PathGraphQ[CycleGraph[2]]").unwrap(), "False");
    assert_eq!(
      interpret("PathGraphQ[Graph[{3, 1, 2}, {3 <-> 1, 1 <-> 2}]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("PathGraphQ[x]").unwrap(), "False");
  }

  #[test]
  fn empty_graph_q() {
    assert_eq!(interpret("EmptyGraphQ[Graph[{1, 2}, {}]]").unwrap(), "True");
    assert_eq!(interpret("EmptyGraphQ[Graph[{}, {}]]").unwrap(), "True");
    assert_eq!(interpret("EmptyGraphQ[CycleGraph[3]]").unwrap(), "False");
    assert_eq!(interpret("EmptyGraphQ[x]").unwrap(), "False");
  }
}

mod planar_graph_q {
  use super::*;

  #[test]
  fn classic_planar_and_nonplanar() {
    assert_eq!(interpret("PlanarGraphQ[CompleteGraph[4]]").unwrap(), "True");
    assert_eq!(
      interpret("PlanarGraphQ[CompleteGraph[5]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PlanarGraphQ[CompleteGraph[6]]").unwrap(),
      "False"
    );
    // K3,3
    assert_eq!(
      interpret(
        "PlanarGraphQ[Graph[{1, 2, 3, 4, 5, 6}, {1 <-> 4, 1 <-> 5, 1 <-> 6, 2 <-> 4, 2 <-> 5, 2 <-> 6, 3 <-> 4, 3 <-> 5, 3 <-> 6}]]"
      )
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PlanarGraphQ[PetersenGraph[5, 2]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PlanarGraphQ[GridGraph[{4, 4}]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("PlanarGraphQ[WheelGraph[8]]").unwrap(), "True");
  }

  #[test]
  fn boundary_cases() {
    // K5 minus one edge is planar
    assert_eq!(
      interpret(
        "PlanarGraphQ[Graph[Range[5], {1 <-> 2, 1 <-> 3, 1 <-> 4, 1 <-> 5, 2 <-> 3, 2 <-> 4, 2 <-> 5, 3 <-> 4, 3 <-> 5}]]"
      )
      .unwrap(),
      "True"
    );
    // The cube graph is planar; adding a crossing chord breaks it
    assert_eq!(
      interpret(
        "PlanarGraphQ[Graph[Range[8], {1 <-> 2, 2 <-> 3, 3 <-> 4, 4 <-> 1, 5 <-> 6, 6 <-> 7, 7 <-> 8, 8 <-> 5, 1 <-> 5, 2 <-> 6, 3 <-> 7, 4 <-> 8}]]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "PlanarGraphQ[Graph[Range[8], {1 <-> 2, 2 <-> 3, 3 <-> 4, 4 <-> 1, 5 <-> 6, 6 <-> 7, 7 <-> 8, 8 <-> 5, 1 <-> 5, 2 <-> 6, 3 <-> 7, 4 <-> 8, 1 <-> 7}]]"
      )
      .unwrap(),
      "False"
    );
    // Harary graphs: H(4, n) is planar for some n, not others
    assert_eq!(
      interpret("PlanarGraphQ[HararyGraph[4, 7]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PlanarGraphQ[HararyGraph[4, 8]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PlanarGraphQ[HararyGraph[2, 9]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn decomposition_handles_components() {
    // A disconnected graph containing K3,3 plus a path
    assert_eq!(
      interpret(
        "PlanarGraphQ[Graph[Range[9], {1 <-> 4, 1 <-> 5, 1 <-> 6, 2 <-> 4, 2 <-> 5, 2 <-> 6, 3 <-> 4, 3 <-> 5, 3 <-> 6, 7 <-> 8, 8 <-> 9}]]"
      )
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PlanarGraphQ[Graph[{1, 2, 3, 4}, {1 <-> 2, 3 <-> 4}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn degenerate_graphs_are_planar() {
    assert_eq!(interpret("PlanarGraphQ[Graph[{}, {}]]").unwrap(), "True");
    assert_eq!(interpret("PlanarGraphQ[Graph[{1}, {}]]").unwrap(), "True");
    // Self-loops and doubled edges never affect planarity
    assert_eq!(interpret("PlanarGraphQ[CycleGraph[1]]").unwrap(), "True");
    assert_eq!(interpret("PlanarGraphQ[CycleGraph[2]]").unwrap(), "True");
    assert_eq!(interpret("PlanarGraphQ[x]").unwrap(), "False");
  }
}

mod graph_metrics {
  use super::*;

  #[test]
  fn global_clustering_coefficient() {
    // Triangle with a pendant: 3 triangles-counted / 5 connected triples
    assert_eq!(
      interpret(
        "GlobalClusteringCoefficient[Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]]"
      )
      .unwrap(),
      "3/5"
    );
    assert_eq!(
      interpret("GlobalClusteringCoefficient[CompleteGraph[5]]").unwrap(),
      "1"
    );
    // Triangle-free graphs give 0
    assert_eq!(
      interpret("GlobalClusteringCoefficient[CycleGraph[5]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("GlobalClusteringCoefficient[PetersenGraph[5, 2]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("GlobalClusteringCoefficient[WheelGraph[6]]").unwrap(),
      "3/5"
    );
    // No connected triples at all also gives 0
    assert_eq!(
      interpret("GlobalClusteringCoefficient[Graph[{1}, {}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn mean_clustering_coefficient() {
    // (1 + 1 + 1/3 + 0)/4
    assert_eq!(
      interpret(
        "MeanClusteringCoefficient[Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]]"
      )
      .unwrap(),
      "7/12"
    );
    assert_eq!(
      interpret("MeanClusteringCoefficient[CompleteGraph[5]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MeanClusteringCoefficient[WheelGraph[6]]").unwrap(),
      "23/36"
    );
    assert_eq!(
      interpret("MeanClusteringCoefficient[Graph[{1}, {}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn graph_density() {
    assert_eq!(
      interpret(
        "GraphDensity[Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]]"
      )
      .unwrap(),
      "2/3"
    );
    assert_eq!(interpret("GraphDensity[CompleteGraph[5]]").unwrap(), "1");
    assert_eq!(interpret("GraphDensity[CycleGraph[5]]").unwrap(), "1/2");
    assert_eq!(interpret("GraphDensity[StarGraph[7]]").unwrap(), "2/7");
    assert_eq!(interpret("GraphDensity[Graph[{1, 2}, {}]]").unwrap(), "0");
    // Doubled edges count once
    assert_eq!(interpret("GraphDensity[CycleGraph[2]]").unwrap(), "1");
    // The single-vertex formula degenerates: stays unevaluated
    assert_eq!(
      interpret("GraphDensity[Graph[{1}, {}]]").unwrap(),
      "GraphDensity[Graph[<1>, <0>]]"
    );
    assert_eq!(interpret("GraphDensity[x]").unwrap(), "GraphDensity[x]");
  }

  #[test]
  fn mean_graph_distance() {
    assert_eq!(
      interpret(
        "MeanGraphDistance[Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]]"
      )
      .unwrap(),
      "4/3"
    );
    assert_eq!(
      interpret("MeanGraphDistance[CompleteGraph[5]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MeanGraphDistance[CycleGraph[5]]").unwrap(),
      "3/2"
    );
    assert_eq!(
      interpret("MeanGraphDistance[PetersenGraph[5, 2]]").unwrap(),
      "5/3"
    );
    assert_eq!(
      interpret("MeanGraphDistance[GridGraph[{3, 3}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("MeanGraphDistance[PathGraph[Range[6]]]").unwrap(),
      "7/3"
    );
    // Disconnected graphs give Infinity
    assert_eq!(
      interpret("MeanGraphDistance[Graph[{1, 2, 3, 4}, {1 <-> 2, 3 <-> 4}]]")
        .unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("MeanGraphDistance[Graph[{1}, {}]]").unwrap(),
      "MeanGraphDistance[Graph[<1>, <0>]]"
    );
  }
}

mod graph_accessors {
  use super::*;

  #[test]
  fn adjacency_list() {
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]; AdjacencyList[g, 3]"
      )
      .unwrap(),
      "{1, 2, 4}"
    );
    // One-argument form: all neighbor lists
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]; AdjacencyList[g]"
      )
      .unwrap(),
      "{{2, 3}, {1, 3}, {1, 2, 4}, {3}}"
    );
    // Neighbors follow the vertex-list order, not value order
    assert_eq!(
      interpret("AdjacencyList[Graph[{3, 1, 2}, {3 <-> 1, 1 <-> 2}], 1]")
        .unwrap(),
      "{3, 2}"
    );
    assert_eq!(
      interpret("AdjacencyList[PetersenGraph[5, 2], 1]").unwrap(),
      "{3, 4, 6}"
    );
    assert_eq!(
      interpret("AdjacencyList[StarGraph[5], 1]").unwrap(),
      "{2, 3, 4, 5}"
    );
  }

  #[test]
  fn incidence_list() {
    // Incident edges in EdgeList order with stored orientation
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]; IncidenceList[g, 3]"
      )
      .unwrap(),
      "{1 \u{f3d4} 3, 2 \u{f3d4} 3, 3 \u{f3d4} 4}"
    );
    assert_eq!(
      interpret("IncidenceList[CycleGraph[5], 1]").unwrap(),
      "{1 \u{f3d4} 2, 1 \u{f3d4} 5}"
    );
  }

  #[test]
  fn edge_index() {
    // TwoWayRule and UndirectedEdge specs, either endpoint order
    assert_eq!(
      interpret(
        "g = Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]; {EdgeIndex[g, 2 <-> 3], EdgeIndex[g, 3 <-> 2]}"
      )
      .unwrap(),
      "{3, 3}"
    );
    assert_eq!(
      interpret("EdgeIndex[CycleGraph[5], UndirectedEdge[1, 2]]").unwrap(),
      "1"
    );
    assert_eq!(interpret("EdgeIndex[CycleGraph[5], 1 <-> 5]").unwrap(), "2");
  }

  #[test]
  fn invalid_arguments() {
    assert_eq!(
      interpret("AdjacencyList[CycleGraph[5], 9]").unwrap(),
      "AdjacencyList[Graph[<5>, <5>], 9]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "AdjacencyList::inv: The argument 9 in AdjacencyList[Graph[<5>, <5>], 9] is not a valid vertex."
      )),
      "expected inv message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("EdgeIndex[CycleGraph[5], 1 <-> 3]").unwrap(),
      "EdgeIndex[Graph[<5>, <5>], 1 <-> 3]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "EdgeIndex::inv: The argument 1 <-> 3 in EdgeIndex[Graph[<5>, <5>], 1 <-> 3] is not a valid edge."
      )),
      "expected inv message, got {:?}",
      msgs
    );
    // Non-graphs stay silently unevaluated
    assert_eq!(
      interpret("AdjacencyList[x, 1]").unwrap(),
      "AdjacencyList[x, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }
}

mod graph_assortativity {
  use super::*;

  #[test]
  fn newman_coefficient_exact() {
    // Triangle with a pendant, computed by hand: -5/7
    assert_eq!(
      interpret(
        "GraphAssortativity[Graph[{1, 2, 3, 4}, {1 <-> 2, 1 <-> 3, 2 <-> 3, 3 <-> 4}]]"
      )
      .unwrap(),
      "-5/7"
    );
    // Stars are maximally disassortative
    assert_eq!(interpret("GraphAssortativity[StarGraph[5]]").unwrap(), "-1");
    assert_eq!(interpret("GraphAssortativity[StarGraph[7]]").unwrap(), "-1");
    assert_eq!(
      interpret("GraphAssortativity[GridGraph[{2, 3}]]").unwrap(),
      "-1/6"
    );
    assert_eq!(
      interpret("GraphAssortativity[GridGraph[{3, 3}]]").unwrap(),
      "-1/17"
    );
    assert_eq!(
      interpret("GraphAssortativity[PathGraph[Range[5]]]").unwrap(),
      "-1/3"
    );
    assert_eq!(
      interpret("GraphAssortativity[WheelGraph[6]]").unwrap(),
      "-1/3"
    );
  }

  #[test]
  fn regular_graphs_give_zero() {
    assert_eq!(interpret("GraphAssortativity[CycleGraph[5]]").unwrap(), "0");
    assert_eq!(
      interpret("GraphAssortativity[PetersenGraph[5, 2]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("GraphAssortativity[Graph[{1, 2}, {1 <-> 2}]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("GraphAssortativity[HararyGraph[3, 8]]").unwrap(),
      "0"
    );
    // Edgeless graphs too
    assert_eq!(
      interpret("GraphAssortativity[Graph[{1}, {}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn non_graph_stays_unevaluated() {
    assert_eq!(
      interpret("GraphAssortativity[x]").unwrap(),
      "GraphAssortativity[x]"
    );
  }
}
