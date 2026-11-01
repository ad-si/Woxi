// Argument forms of the graph, image and audio functions found missing by the
// reference-page parameter sweep (round 5). Expected values are the ones the
// Wolfram reference pages show, or follow from their documented semantics.

use super::*;

mod graph_functions_accept_edge_rule_lists {
  use super::*;

  #[test]
  fn adjacency_list_of_a_rule_list() {
    // Neighbours in either direction, in the graph's vertex order
    assert_eq!(
      interpret(
        "AdjacencyList[{1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, 5 -> 4, 6 -> 1}, 1]"
      )
      .unwrap(),
      "{3, 2, 6, 5}"
    );
    assert_eq!(
      interpret("AdjacencyList[{1 -> 3, 2 -> 1, 3 -> 6}]").unwrap(),
      "{{3, 2}, {1, 6}, {1}, {3}}"
    );
  }

  #[test]
  fn adjacency_list_of_a_directed_graph() {
    assert_eq!(
      interpret("AdjacencyList[Graph[{1 -> 2, 3 -> 1, 2 -> 3}], 2]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn incidence_list_and_edge_index_of_directed_edges() {
    assert_eq!(
      interpret(
        "e = IncidenceList[{1 -> 3, 2 -> 1, 3 -> 6}, 1]; {Head /@ e, List @@@ e}"
      )
      .unwrap(),
      "{{DirectedEdge, DirectedEdge}, {{1, 3}, {2, 1}}}"
    );
    assert_eq!(
      interpret("EdgeIndex[Graph[{1 -> 3, 2 -> 1, 3 -> 6}], 2 -> 1]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret(
        "EdgeIndex[Graph[{1 -> 3, 2 -> 1, 3 -> 6}], DirectedEdge[3, 6]]"
      )
      .unwrap(),
      "3"
    );
    // A directed edge only matches in its own orientation
    assert_eq!(
      interpret("EdgeIndex[Graph[{1 -> 3, 2 -> 1, 3 -> 6}], 1 -> 2]").unwrap(),
      "EdgeIndex[Graph[<4>, <3>], 1 -> 2]"
    );
    assert_eq!(interpret("EdgeIndex[CycleGraph[4], 2 <-> 1]").unwrap(), "1");
  }

  #[test]
  fn vertex_replace_of_a_rule_list() {
    assert_eq!(
      interpret(
        "e = EdgeList[VertexReplace[{1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, 5 -> 4, 6 -> 1}, {1 -> \"hi\"}]]; \
         {Union[Head /@ e], List @@@ e}"
      )
      .unwrap(),
      "{{DirectedEdge}, {{hi, 3}, {2, hi}, {3, 6}, {4, 6}, {hi, 5}, {5, 4}, {6, hi}}}"
    );
  }

  #[test]
  fn vertex_eccentricity_of_a_rule_list() {
    assert_eq!(
      interpret(
        "VertexEccentricity[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4, 4 -> 5, 5 -> 3}, 1]"
      )
      .unwrap(),
      "4"
    );
  }

  #[test]
  fn find_minimum_cost_flow_of_a_rule_list() {
    assert_eq!(
      interpret("FindMinimumCostFlow[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}, 1, 4]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "FindMinimumCostFlow[Graph[{\"a\" -> \"b\", \"b\" -> \"c\", \"a\" -> \"c\"}], \"a\", \"c\"]"
      )
      .unwrap(),
      "3"
    );
    // No path from 4 back to 1
    assert_eq!(
      interpret("FindMinimumCostFlow[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}, 4, 1]")
        .unwrap(),
      "FindMinimumCostFlow[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}, 4, 1]"
    );
  }
}

mod random_graph_counts {
  use super::*;

  #[test]
  fn an_array_of_graphs() {
    assert_eq!(
      interpret(
        "Map[VertexCount, RandomGraph[BernoulliGraphDistribution[8, 0.3], {2, 3}], {2}]"
      )
      .unwrap(),
      "{{8, 8, 8}, {8, 8, 8}}"
    );
    assert_eq!(
      interpret(
        "Dimensions[Map[VertexCount, RandomGraph[SpatialGraphDistribution[15, 0.35], {2, 2}], {2}]]"
      )
      .unwrap(),
      "{2, 2}"
    );
  }

  #[test]
  fn a_list_of_graphs() {
    assert_eq!(
      interpret(
        "VertexCount /@ RandomGraph[DegreeGraphDistribution[{3, 3, 3, 3, 3, 3, 3, 3}], 4]"
      )
      .unwrap(),
      "{8, 8, 8, 8}"
    );
    assert_eq!(interpret("Length[RandomGraph[{5, 10}, {1}]]").unwrap(), "1");
  }

  #[test]
  fn invalid_shape_stays_unevaluated() {
    assert_eq!(
      interpret("RandomGraph[{5, 10}, {2, x}]").unwrap(),
      "RandomGraph[{5, 10}, {2, x}]"
    );
  }
}

mod price_graph_distribution {
  use super::*;

  #[test]
  fn directed_with_documented_edge_count() {
    // EdgeCount is k (2 n - 1 - k) / 2
    assert_eq!(
      interpret(
        "g = RandomGraph[PriceGraphDistribution[15, 2, 1]]; \
         {VertexCount[g], EdgeCount[g], DirectedGraphQ[g], SimpleGraphQ[g]}"
      )
      .unwrap(),
      "{15, 27, True, True}"
    );
    assert_eq!(
      interpret(
        "Union[EdgeCount /@ RandomGraph[PriceGraphDistribution[7, 3, 1.5], 20]]"
      )
      .unwrap(),
      "{15}"
    );
    assert_eq!(
      interpret("Length[RandomGraph[PriceGraphDistribution[15, 2, 1], 4]]")
        .unwrap(),
      "4"
    );
  }

  #[test]
  fn undirected_option() {
    assert_eq!(
      interpret(
        "DirectedGraphQ[RandomGraph[PriceGraphDistribution[5, 2, 1.5, DirectedEdges -> False]]]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn edges_point_from_new_to_old_vertices() {
    assert_eq!(
      interpret(
        "AllTrue[EdgeList[RandomGraph[PriceGraphDistribution[20, 2, 1]]], \
         First[#] > Last[#] &]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn stays_symbolic() {
    assert_eq!(
      interpret("PriceGraphDistribution[15, 2, 1]").unwrap(),
      "PriceGraphDistribution[15, 2, 1]"
    );
  }
}

mod graphics_grid_of_graphs {
  use super::*;

  #[test]
  fn ragged_upto_rows_of_tooltip_graphs() {
    assert_eq!(
      interpret(
        "Head[GraphicsGrid[Partition[Tooltip[CycleGraph[#], \"c\"] & /@ Range[3, 7], UpTo[3]]]]"
      )
      .unwrap(),
      "Graphics"
    );
    assert_eq!(
      interpret(
        "Head[GraphicsRow[VertexDelete[CycleGraph[6], #] & /@ {1, 2}, Dividers -> All]]"
      )
      .unwrap(),
      "Graphics"
    );
  }
}

mod image_crop_size_specs {
  use super::*;

  const IMG: &str = "i = Image[RandomReal[1, {200, 300, 3}]];";

  fn dims(spec: &str) -> String {
    interpret(&format!("{IMG} ImageDimensions[ImageCrop[i, {spec}]]")).unwrap()
  }

  #[test]
  fn full_keeps_a_dimension() {
    assert_eq!(dims("{100, Full}"), "{100, 200}");
    assert_eq!(dims("{Full, Full}"), "{300, 200}");
  }

  #[test]
  fn automatic_keeps_the_aspect_ratio() {
    assert_eq!(dims("{Automatic, 100}"), "{150, 100}");
    assert_eq!(dims("{150, Automatic}"), "{150, 100}");
  }

  #[test]
  fn upto_never_grows_the_image() {
    assert_eq!(dims("UpTo[100]"), "{100, 100}");
    assert_eq!(dims("UpTo[250]"), "{250, 200}");
    assert_eq!(dims("{UpTo[500], UpTo[100]}"), "{300, 100}");
    assert_eq!(
      interpret(&format!(
        "{IMG} ImageDimensions[ImageCrop[ImageRotate[i], UpTo[100]]]"
      ))
      .unwrap(),
      "{100, 100}"
    );
  }

  #[test]
  fn range_clamps_the_extent() {
    assert_eq!(dims("{{50, 80}, 100}"), "{80, 100}");
    assert_eq!(dims("{{400, 500}, 100}"), "{400, 100}");
  }

  #[test]
  fn padding_none_never_adds_pixels() {
    assert_eq!(dims("{400, 100}, Padding -> None"), "{300, 100}");
    assert_eq!(dims("{400, 100}"), "{400, 100}");
  }

  #[test]
  fn plain_sizes_still_crop_centered() {
    assert_eq!(
      interpret(&format!(
        "{IMG} ImageCrop[i, {{100, 50}}] == ImageTake[i, {{1, -1}} ((200 - 50)/2 + 1), {{1, -1}} ((300 - 100)/2 + 1)]"
      ))
      .unwrap(),
      "True"
    );
  }
}

mod image_take_3d {
  use super::*;

  const IMG: &str = "i = Image3D[RandomReal[1, {30, 40, 50}]];";

  fn dims(spec: &str) -> String {
    interpret(&format!("{IMG} ImageDimensions[ImageTake[i, {spec}]]")).unwrap()
  }

  #[test]
  fn slices_rows_and_columns() {
    assert_eq!(dims("All, -23, -23"), "{23, 23, 30}");
    assert_eq!(dims("{1, 20}, {26, 40}, {26, 46}"), "{21, 15, 20}");
  }

  #[test]
  fn slices_only() {
    assert_eq!(dims("-23"), "{50, 40, 23}");
    assert_eq!(dims("{15}"), "{50, 40, 1}");
  }

  #[test]
  fn takes_the_right_voxel() {
    assert_eq!(
      interpret(&format!(
        "{IMG} ImageData[ImageTake[i, {{2}}, {{3}}, {{4}}]] == {{{{{{ImageData[i][[2, 3, 4]]}}}}}}"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn four_arguments_need_a_3d_image() {
    assert_eq!(
      interpret("Head[ImageTake[Image[RandomReal[1, {3, 3}]], 1, 1, 1]]")
        .unwrap(),
      "ImageTake"
    );
  }
}

mod image_collage_weights {
  use super::*;

  const IMGS: &str = "i = Image[RandomReal[1, {20, 30, 3}]]; \
                      j = Image[RandomReal[1, {40, 40, 3}]];";

  fn head(expr: &str) -> String {
    interpret(&format!("{IMGS} Head[{expr}]")).unwrap()
  }

  #[test]
  fn weight_list_rule() {
    assert_eq!(head("ImageCollage[{1, 1, 4, 9} -> {i, i, j, i}]"), "Image");
    assert_eq!(head("ImageCollage[{1, 2} -> {i, j}, \"Fill\"]"), "Image");
  }

  #[test]
  fn scaled_weights() {
    assert_eq!(
      head("ImageCollage[Scaled[Range[4]] -> {i, i, j, i}]"),
      "Image"
    );
    assert_eq!(head("ImageCollage[Scaled[1] -> {i, j}]"), "Image");
  }

  #[test]
  fn shared_weight_rule_list_and_association() {
    assert_eq!(head("ImageCollage[2 -> {i, j}]"), "Image");
    assert_eq!(head("ImageCollage[{1 -> i, 2 -> j}]"), "Image");
    assert_eq!(head("ImageCollage[<|i -> 1, j -> 3|>]"), "Image");
  }
}

mod audio {
  use super::*;

  const AUDIO: &str = "a = Audio[Table[Sin[2 Pi 440 t/8000.], {t, 0, 3999}], SampleRate -> 8000];";

  fn local(rest: &str) -> String {
    interpret(&format!(
      "{AUDIO} AudioLocalMeasurements[a, {{\"RMSAmplitude\", \"Power\"}}{rest}"
    ))
    .unwrap()
  }

  #[test]
  fn output_formats() {
    assert_eq!(local(", \"Association\"] // Keys"), "{RMSAmplitude, Power}");
    assert_eq!(local(", \"RuleList\"] // Keys"), "{RMSAmplitude, Power}");
    assert_eq!(local(", \"List\"] // Length"), "2");
    assert_eq!(local(", \"Dataset\"] // Head"), "Dataset");
    assert_eq!(local(", Automatic] // Length"), "2");
    assert_eq!(local("] // Length"), "2");
  }

  #[test]
  fn single_property_formats() {
    assert_eq!(
      interpret(&format!(
        "{AUDIO} Length[AudioLocalMeasurements[a, \"Power\", \"List\"]]"
      ))
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret(&format!(
        "{AUDIO} Keys[AudioLocalMeasurements[a, \"Power\", \"Association\"]]"
      ))
      .unwrap(),
      "{Power}"
    );
  }

  #[test]
  fn unknown_format_stays_unevaluated() {
    assert_eq!(
      interpret(&format!(
        "{AUDIO} Head[AudioLocalMeasurements[a, \"Power\", \"Bogus\"]]"
      ))
      .unwrap(),
      "AudioLocalMeasurements"
    );
  }

  #[test]
  fn a_one_second_tone_builds_quickly() {
    // Exact samples Sin[2 Pi 1200 k/44099] used to take ~25 s to build.
    let start = std::time::Instant::now();
    assert_eq!(
      interpret(
        "QuantityMagnitude[AudioLength[Audio[Array[Sin[2 Pi 1200 #] &, 44100, {0, 1}]]]]"
      )
        .unwrap(),
      "44100"
    );
    assert!(start.elapsed().as_secs() < 20);
  }
}

mod exact_trig_beyond_the_first_octant {
  use super::*;

  #[test]
  fn folds_to_the_first_octant() {
    assert_eq!(
      interpret("Sin[400000 Pi/44099]").unwrap(),
      "-Sin[(3109*Pi)/44099]"
    );
    assert_eq!(interpret("Cos[5 Pi/7]").unwrap(), "-Sin[(3*Pi)/14]");
    assert_eq!(interpret("Tan[3 Pi/7]").unwrap(), "Cot[Pi/14]");
    assert_eq!(interpret("Sec[3 Pi/7]").unwrap(), "Csc[Pi/14]");
  }
}

mod array_over_an_exact_range {
  use super::*;

  #[test]
  fn rational_samples() {
    assert_eq!(
      interpret("Array[f, 5, {0, 1}]").unwrap(),
      "{f[0], f[1/4], f[1/2], f[3/4], f[1]}"
    );
    assert_eq!(
      interpret("Array[f, 4, {-1/2, 3}]").unwrap(),
      "{f[-1/2], f[2/3], f[11/6], f[3]}"
    );
    assert_eq!(
      interpret("Array[f, 3, {2, -2}]").unwrap(),
      "{f[2], f[0], f[-2]}"
    );
    assert_eq!(
      interpret("Array[f, 3, {0., 1}]").unwrap(),
      "{f[0.], f[0.5], f[1.]}"
    );
  }
}
