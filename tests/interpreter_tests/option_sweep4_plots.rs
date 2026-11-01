// Argument forms of the plot and chart functions found missing by the
// reference-page parameter sweep (round 4). Graphics are checked by their
// structure — primitives, labels, colours — not by their full SVG text.

use super::*;

/// The SVG `ExportString` draws for `expr`.
fn svg(expr: &str) -> String {
  let out = interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap();
  assert!(out.starts_with("<svg"), "{expr} gave no picture: {out}");
  out
}

/// The text of every `<text>` element of an SVG, in document order.
fn texts(svg: &str) -> Vec<String> {
  svg
    .split("<text")
    .skip(1)
    .filter_map(|t| {
      let start = t.find('>')? + 1;
      let end = t.find("</text>")?;
      Some(t[start..end].trim().to_string())
    })
    .collect()
}

fn has_text(svg: &str, label: &str) -> bool {
  texts(svg).iter().any(|t| t == label)
}

/// How many times `fill="rgb(r,g,b)"` occurs.
fn fills(svg: &str, rgb: &str) -> usize {
  svg.matches(&format!("fill=\"rgb({rgb})\"")).count()
}

mod chart_association_data {
  use super::*;

  #[test]
  fn bar_chart_plots_the_values() {
    assert_eq!(
      interpret("BarChart[<|\"a\" -> 1, \"b\" -> 2, \"c\" -> 5, \"d\" -> 3|>]")
        .unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      svg("BarChart[<|\"a\" -> 1, \"b\" -> 2, \"c\" -> 5, \"d\" -> 3|>]"),
      svg("BarChart[{1, 2, 5, 3}]")
    );
  }

  #[test]
  fn plot_labels_automatic_uses_the_keys() {
    let s = svg(
      "BarChart[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
       PlotLabels -> Automatic]",
    );
    for k in ["ka", "kb", "kc"] {
      assert!(has_text(&s, k), "missing {k}: {:?}", texts(&s));
    }
    // The old option name does the same.
    assert_eq!(
      s,
      svg(
        "BarChart[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
         ChartLabels -> Automatic]"
      )
    );
  }

  #[test]
  fn callout_labels_use_the_keys() {
    let s = svg(
      "BarChart[<|\"ka\" -> 1, \"kb\" -> 2|>, \
       PlotLabels -> Callout[Automatic, Above]]",
    );
    assert!(has_text(&s, "ka") && has_text(&s, "kb"), "{:?}", texts(&s));
  }

  #[test]
  fn nested_associations_group_bars_and_legend_the_inner_keys() {
    let s = svg(
      "BarChart[<|\"ga\" -> <|\"ea\" -> 1, \"eb\" -> 2|>, \
       \"gb\" -> <|\"ea\" -> 4, \"eb\" -> 1|>|>, PlotLegends -> Automatic]",
    );
    assert!(has_text(&s, "ea") && has_text(&s, "eb"), "{:?}", texts(&s));
    assert!(!has_text(&s, "ga"), "{:?}", texts(&s));
    assert_eq!(
      s,
      svg("BarChart[{{1, 2}, {4, 1}}, ChartLegends -> {\"ea\", \"eb\"}]")
    );
  }

  #[test]
  fn plot_style_is_the_bar_style() {
    assert_eq!(
      svg("BarChart[{1, 2}, PlotStyle -> {Red, Blue}]"),
      svg("BarChart[{1, 2}, ChartStyle -> {Red, Blue}]")
    );
  }

  #[test]
  fn pie_chart_labels_and_legends_from_keys() {
    let labelled = svg(
      "PieChart[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
       PlotLabels -> Automatic]",
    );
    assert!(has_text(&labelled, "ka"), "{:?}", texts(&labelled));
    let called_out = svg(
      "PieChart[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
       PlotLabels -> Callout[Automatic]]",
    );
    assert!(has_text(&called_out, "kc"), "{:?}", texts(&called_out));
    let legended = svg(
      "PieChart[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
       PlotLegends -> Automatic]",
    );
    for k in ["ka", "kb", "kc"] {
      assert!(
        has_text(&legended, k),
        "missing {k}: {:?}",
        texts(&legended)
      );
    }
    // One swatch per sector beside the pie.
    assert_eq!(legended.matches("<rect").count(), 3 + 1, "{legended}");
  }

  #[test]
  fn pie_chart_legends_list() {
    let s = svg("PieChart[{1, 2}, ChartLegends -> {\"la\", \"lb\"}]");
    assert!(has_text(&s, "la") && has_text(&s, "lb"), "{:?}", texts(&s));
  }

  #[test]
  fn histogram_of_association_values() {
    assert_eq!(
      svg(
        "Histogram[<|\"a\" -> 2, \"b\" -> 3, \"c\" -> 5, \"d\" -> 7, \
         \"e\" -> 11, \"f\" -> 13|>, 5]"
      ),
      svg("Histogram[{2, 3, 5, 7, 11, 13}, 5]")
    );
  }

  #[test]
  fn histogram_of_nested_association_is_two_datasets() {
    assert_eq!(
      svg(
        "Histogram[<|\"one\" -> <|\"a\" -> 6, \"b\" -> 3, \"c\" -> 5|>, \
         \"two\" -> <|\"a\" -> 26, \"b\" -> 23, \"c\" -> 32|>|>, 6]"
      ),
      svg("Histogram[{{6, 3, 5}, {26, 23, 32}}, 6]")
    );
  }

  #[test]
  fn histogram_legends_name_the_datasets() {
    let s = svg(
      "Histogram[<|\"data one\" -> {6, 3, 5, 7, 11, 13}, \
       \"data two\" -> {26, 23, 32, 29, 21, 38}|>, 6, PlotLegends -> Automatic]",
    );
    assert!(
      has_text(&s, "data one") && has_text(&s, "data two"),
      "{:?}",
      texts(&s)
    );
    let listed = svg(
      "Histogram[{{1, 2, 3}, {2, 3, 4}}, ChartLegends -> {\"la\", \"lb\"}]",
    );
    assert!(has_text(&listed, "la") && has_text(&listed, "lb"));
  }

  #[test]
  fn time_series_values_ignore_the_time_stamps() {
    assert_eq!(
      svg("BarChart[TimeSeries[{19, 16, 9, 3}, {\"May 24, 1982\"}]]"),
      svg("BarChart[{19, 16, 9, 3}]")
    );
    assert_eq!(
      svg("BarChart[TimeSeries[{{1, 19}, {2, 16}, {5, 9}}]]"),
      svg("BarChart[{19, 16, 9}]")
    );
    assert_eq!(
      svg("PieChart[TimeSeries[{3, 1, 2}, {1}]]"),
      svg("PieChart[{3, 1, 2}]")
    );
  }
}

mod labeled_list_data {
  use super::*;

  #[test]
  fn list_line_plot_rule_labels_the_points() {
    let s =
      svg("ListLinePlot[{2, 3, 5, 7} -> {\"pa\", \"pb\", \"pc\", \"pd\"}]");
    for k in ["pa", "pb", "pc", "pd"] {
      assert!(has_text(&s, k), "missing {k}: {:?}", texts(&s));
    }
    assert!(
      s.contains("<polyline") || s.contains("<path"),
      "no line: {s}"
    );
  }

  #[test]
  fn labeling_function_position_places_the_labels() {
    let label_y = |s: &str| -> f64 {
      let i = s.find(">pa").unwrap();
      let tag = &s[s[..i].rfind("<text").unwrap()..i];
      let y = tag.split("y=\"").nth(1).unwrap();
      y[..y.find('"').unwrap()].parse().unwrap()
    };
    let below = svg(
      "ListLinePlot[{2, 3, 5} -> {\"pa\", \"pb\", \"pc\"}, \
       LabelingFunction -> Below]",
    );
    let above = svg(
      "ListLinePlot[{2, 3, 5} -> {\"pa\", \"pb\", \"pc\"}, \
       LabelingFunction -> Above]",
    );
    assert!(label_y(&below) > label_y(&above));
    assert!(below.contains("text-anchor=\"middle\">pa"), "{below}");
  }

  #[test]
  fn list_plot_rule_with_one_label() {
    let s = svg("ListPlot[{1, 2, 3} -> \"tree 1\"]");
    assert!(has_text(&s, "tree 1"), "{:?}", texts(&s));
  }
}

mod complex_list_plot_data_forms {
  use super::*;

  #[test]
  fn labels_from_a_rule() {
    let s = svg(
      "ComplexListPlot[{1 + I, 2 + 2 I, 3 + I} -> {\"za\", \"zb\", \"zc\"}, \
       LabelingFunction -> Above]",
    );
    for k in ["za", "zb", "zc"] {
      assert!(has_text(&s, k), "missing {k}: {:?}", texts(&s));
    }
    assert_eq!(
      s,
      svg(
        "ListPlot[{{1, 1}, {2, 2}, {3, 1}} -> {\"za\", \"zb\", \"zc\"}, \
         LabelingFunction -> Above]"
      )
    );
  }

  #[test]
  fn association_and_sparse_array_data() {
    assert_eq!(
      interpret("ComplexListPlot[<|\"a\" -> 1 + I, \"b\" -> 2 - I|>]").unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      svg("ComplexListPlot[SparseArray[{1 -> 1 + I, 3 -> 2 I}]]"),
      svg("ComplexListPlot[{1 + I, 0, 2 I}]")
    );
    assert_eq!(
      interpret("ComplexListPlot[Tooltip[{1, I, -1}]]").unwrap(),
      "-Graphics-"
    );
  }
}

mod number_line_plot_data_forms {
  use super::*;

  #[test]
  fn rule_labels_the_points() {
    let s = svg("NumberLinePlot[{1, 2, 3} -> {\"na\", \"nb\", \"nc\"}]");
    for k in ["na", "nb", "nc"] {
      assert!(has_text(&s, k), "missing {k}: {:?}", texts(&s));
    }
    assert_eq!(s.matches("<circle").count(), 3);
  }

  #[test]
  fn association_values() {
    assert_eq!(
      svg("NumberLinePlot[<|\"a\" -> 1, \"b\" -> 3|>]"),
      svg("NumberLinePlot[{1, 3}]")
    );
    assert_eq!(
      svg("NumberLinePlot[<|\"a\" -> {1, 2}, \"b\" -> {3}|>]"),
      svg("NumberLinePlot[{{1, 2}, {3}}]")
    );
  }
}

mod list_point_plot_3d_datasets {
  use super::*;

  #[test]
  fn several_height_matrices() {
    let s = svg(
      "ListPointPlot3D[{Table[Sin[j/3], {i, 4}, {j, 4}], \
       Table[-Sin[j/3], {i, 4}, {j, 4}]}]",
    );
    // 16 points per dataset, each dataset in its own colour.
    assert_eq!(fills(&s, "68,114,196"), 16, "{s}");
    assert_eq!(fills(&s, "237,125,49"), 16, "{s}");
  }

  #[test]
  fn several_point_lists_and_single_forms_still_work() {
    let s = svg("ListPointPlot3D[{{{1, 2, 3}, {2, 3, 4}}, {{0, 0, 0}}}]");
    assert_eq!(fills(&s, "68,114,196"), 2);
    assert_eq!(fills(&s, "237,125,49"), 1);
    let one = svg("ListPointPlot3D[{{1, 2, 3}, {2, 3, 4}, {3, 1, 2}}]");
    assert_eq!(fills(&one, "68,114,196"), 3);
    let matrix = svg("ListPointPlot3D[{{1, 2}, {3, 4}}]");
    assert_eq!(fills(&matrix, "68,114,196"), 4);
  }

  #[test]
  fn rule_labels_the_points() {
    let s = svg(
      "ListPointPlot3D[{{1, 2, 3}, {2, 3, 4}, {3, 1, 2}} -> \
       {\"qa\", \"qb\", \"qc\"}, LabelingFunction -> Callout]",
    );
    for k in ["qa", "qb", "qc"] {
      assert!(has_text(&s, k), "missing {k}: {:?}", texts(&s));
    }
    let hidden = svg(
      "ListPointPlot3D[{{1, 2, 3}, {2, 3, 4}} -> {\"qa\", \"qb\"}, \
       LabelingFunction -> None]",
    );
    assert!(!has_text(&hidden, "qa"));
  }
}

mod list_plot_3d_datasets {
  use super::*;

  /// Facets whose colour is mostly green / yellow.
  fn green_and_yellow(s: &str) -> (usize, usize) {
    let (mut green, mut yellow) = (0, 0);
    for part in s.split("fill=\"rgb(").skip(1) {
      let rgb: Vec<u32> = part[..part.find(')').unwrap()]
        .split(',')
        .map(|c| c.parse().unwrap())
        .collect();
      if rgb[0] < 60 && rgb[1] > 100 && rgb[2] < 60 {
        green += 1;
      } else if rgb[0] > 100 && rgb[1] > 100 && rgb[2] < 60 {
        yellow += 1;
      }
    }
    (green, yellow)
  }

  #[test]
  fn several_surfaces_take_the_plot_styles() {
    let s = svg(
      "ListPlot3D[Table[a Sin[x y], {a, {-1, 1}}, {x, 0, 3, 0.5}, \
       {y, 0, 3, 0.5}], PlotStyle -> {Green, Yellow}, Mesh -> None]",
    );
    let (green, yellow) = green_and_yellow(&s);
    assert!(green >= 36 && yellow >= 36, "{green} {yellow}");
    assert!(s.contains("<text"), "the combined plot keeps its axes: {s}");
  }

  #[test]
  fn single_surface_plot_style() {
    let s = svg(
      "ListPlot3D[Table[Sin[j/3], {i, 5}, {j, 5}], PlotStyle -> Green, \
       Mesh -> None]",
    );
    assert_eq!(green_and_yellow(&s), (32, 0));
  }
}

mod discrete_plot_3d_forms {
  use super::*;

  #[test]
  fn points_with_stems_by_default() {
    let r = interpret(
      "g = DiscretePlot3D[i + j, {i, 0, 2}, {j, 0, 2}]; \
       {Head[g], Count[g, _Point, Infinity], \
        Length[First[Cases[g, Point[p_] :> p, Infinity]]], \
        Length[First[Cases[g, Line[l_] :> l, Infinity]]]}",
    )
    .unwrap();
    assert_eq!(r, "{Graphics3D, 1, 9, 9}");
  }

  #[test]
  fn several_functions_and_table_iterators() {
    assert_eq!(
      interpret(
        "Count[DiscretePlot3D[{i + j, i - j}, {i, {1, 3, 7}}, {j, 0, 4, 2}], \
         _Point, Infinity]"
      )
      .unwrap(),
      "2"
    );
    // The second iterator may depend on the first.
    assert_eq!(
      interpret(
        "Length[First[Cases[DiscretePlot3D[i + j, {i, 0, 2}, {j, 0, i}], \
         Point[p_] :> p, Infinity]]]"
      )
      .unwrap(),
      "6"
    );
    assert_eq!(
      interpret(
        "Length[First[Cases[DiscretePlot3D[i j, {i, 3}, {j, 2}], \
         Point[p_] :> p, Infinity]]]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn extent_size_full_draws_bars_one_step_wide() {
    assert_eq!(
      interpret(
        "c = Cases[DiscretePlot3D[i + j, {i, 0, 2, 2}, {j, 1, 2}, \
         ExtentSize -> Full], _Cuboid, Infinity]; \
         {Length[c], c[[-1]]}"
      )
      .unwrap(),
      "{4, Cuboid[{1., 1.5, 1.}, {3., 2.5, 4.}]}"
    );
    assert_eq!(
      interpret(
        "Cases[DiscretePlot3D[i, {i, 1, 2}, {j, 1, 1}, ExtentSize -> Right], \
         _Cuboid, Infinity][[-1]]"
      )
      .unwrap(),
      "Cuboid[{2., 1., 1.}, {3., 2., 2.}]"
    );
  }

  #[test]
  fn style_wrappers_colour_their_function() {
    assert_eq!(
      interpret(
        "Cases[DiscretePlot3D[{i + j, Style[i - j, Green]}, {i, 2}, {j, 2}], \
         {c_, PointSize[_], _Point, ___} :> c, Infinity]"
      )
      .unwrap(),
      "{RGBColor[0.3686274509803922, 0.5058823529411764, 0.7098039215686275], \
       RGBColor[0, 1, 0]}"
    );
  }

  #[test]
  fn filling_none_drops_the_stems() {
    assert_eq!(
      interpret(
        "Count[DiscretePlot3D[i + j, {i, 2}, {j, 2}, Filling -> None], \
         _Line, Infinity]"
      )
      .unwrap(),
      "0"
    );
  }
}

mod parametric_plot_3d_legended_curves {
  use super::*;

  #[test]
  fn legended_curves_in_a_list_are_drawn() {
    let s = svg(
      "ParametricPlot3D[{{Cos[u], 2 Cos[u], Sin[u]}, \
       Legended[{Cos[u], Cos[u], 2 Sin[u]}, \"label\"], \
       {2 Cos[u], Cos[u], Sin[u]}}, {u, 0, 2 Pi}]",
    );
    for rgb in ["94,129,181", "224,147,44", "143,176,50"] {
      assert!(
        s.contains(&format!("stroke=\"rgb({rgb})\"")),
        "curve in {rgb} missing"
      );
    }
  }
}

mod plot_3d_infinite_ranges {
  use super::*;

  #[test]
  fn infinite_ranges_are_compactified() {
    assert_eq!(
      interpret(
        "Head[Plot3D[ArcTan[x] ArcTan[y], {x, -Infinity, Infinity}, \
         {y, -Infinity, Infinity}]]"
      )
      .unwrap(),
      "Graphics3D"
    );
    // A half-infinite range is folded the same way as in Plot.
    assert_eq!(
      interpret("Head[Plot3D[Exp[-x - y^2], {x, 0, Infinity}, {y, -1, 1}]]")
        .unwrap(),
      "Graphics3D"
    );
  }
}

mod matrix_plot_non_matrix {
  use super::*;

  #[test]
  fn stays_unevaluated_as_matrix_plot() {
    let r =
      interpret_with_stdout("MatrixPlot[m, ColorFunction -> \"Monochrome\"]")
        .unwrap();
    assert_eq!(r.result, "MatrixPlot[m, ColorFunction -> Monochrome]");
    assert_eq!(
      r.warnings,
      vec!["MatrixPlot::mat: Argument m at position 1 is not a list of lists."]
    );
    assert_eq!(
      interpret(
        "Head[MatrixPlot[{{1, 2, 1}, {3, 0, 1}, {0, 0, -1}}, \
         ColorFunction -> \"Monochrome\"]]"
      )
      .unwrap(),
      "Graphics"
    );
  }
}

/// Regression: `Count` and `Position` did not look inside a rendered
/// `Graphics3D` (a plot result), though `Cases` and `Part` did, so
/// `Count[DiscretePlot3D[…], _Point, Infinity]` was 0.
mod count_and_position_in_rendered_graphics {
  use super::*;

  #[test]
  fn count_searches_the_symbolic_form() {
    assert_eq!(
      interpret("Count[Graphics3D[{Point[{1, 2, 3}]}], _Point, Infinity]")
        .unwrap(),
      "1"
    );
  }

  #[test]
  fn position_agrees_with_extract() {
    assert_eq!(
      interpret("Position[Graphics3D[{Point[{1, 2, 3}]}], _Point]").unwrap(),
      "{{1, 1}}"
    );
    assert_eq!(
      interpret(
        "g = Graphics3D[{Point[{1, 2, 3}]}]; Extract[g, Position[g, _Point]]"
      )
      .unwrap(),
      "{Point[{1, 2, 3}]}"
    );
  }
}
