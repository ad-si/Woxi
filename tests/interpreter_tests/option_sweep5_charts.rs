// Argument forms of the remaining charts and 3-D list plots found missing by
// the reference-page parameter sweep (round 5). Graphics are checked by
// their structure — primitives, labels, colours — not by their full SVG text.

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

fn assert_texts(svg: &str, labels: &[&str]) {
  for l in labels {
    assert!(has_text(svg, l), "missing {l}: {:?}", texts(svg));
  }
}

/// How many times `fill="rgb(r,g,b)"` occurs.
fn fills(svg: &str, rgb: &str) -> usize {
  svg.matches(&format!("fill=\"rgb({rgb})\"")).count()
}

/// The width of the SVG's viewBox.
fn view_width(svg: &str) -> f64 {
  let start = svg.find("viewBox=\"0 0 ").unwrap() + "viewBox=\"0 0 ".len();
  let rest = &svg[start..];
  rest[..rest.find(' ').unwrap()].parse().unwrap()
}

const TS_3: &str = "TimeSeries[{{19, 16, 10}, {9, 3, 6}, {7, 6, 2}, {17, 12, 5}}, \
   {\"May 24, 1982\"}]";

mod chart_data_forms {
  use super::*;

  #[test]
  fn bar_chart_3d_time_series_plots_the_values() {
    assert_eq!(
      svg(
        "BarChart3D[TimeSeries[{5, 12, 4, 4, 5, 2, 7}, {\"April 21, 1981\"}]]"
      ),
      svg("BarChart3D[{5, 12, 4, 4, 5, 2, 7}]")
    );
  }

  #[test]
  fn bar_chart_3d_association_plots_the_values() {
    assert_eq!(
      interpret("BarChart3D[<|\"a\" -> 4, \"b\" -> 2, \"c\" -> 6|>]").unwrap(),
      "-Graphics3D-"
    );
    assert_eq!(
      svg("BarChart3D[<|\"a\" -> 4, \"b\" -> 2, \"c\" -> 6|>]"),
      svg("BarChart3D[{4, 2, 6}]")
    );
  }

  #[test]
  fn pie_chart_3d_association_and_time_series() {
    assert_eq!(
      svg("PieChart3D[<|\"a\" -> 1, \"b\" -> 2, \"c\" -> 5, \"d\" -> 3|>]"),
      svg("PieChart3D[{1, 2, 5, 3}]")
    );
    assert_eq!(
      svg("PieChart3D[TimeSeries[{19, 16, 9, 3}, {\"May 24, 1982\"}]]"),
      svg("PieChart3D[{19, 16, 9, 3}]")
    );
  }

  #[test]
  fn weighted_data_weights_are_ignored() {
    assert_eq!(
      svg(
        "BarChart3D[WeightedData[{1, 2, 3, 4, 5}, {0.5, 0.8, 0.2, 0.1, 0.2}]]"
      ),
      svg("BarChart3D[{1, 2, 3, 4, 5}]")
    );
    assert_eq!(
      svg("PieChart3D[WeightedData[{1, 2, 3}, {0.5, 0.2, 0.1}]]"),
      svg("PieChart3D[{1, 2, 3}]")
    );
    assert_eq!(
      svg(
        "SectorChart[WeightedData[{{1, 2}, {3, 4}, {5, 6}}, {0.5, 0.2, 0.1}]]"
      ),
      svg("SectorChart[{{1, 2}, {3, 4}, {5, 6}}]")
    );
    assert_eq!(
      svg("BarChart[WeightedData[{1, 2, 3}, {0.5, 0.8, 0.2}]]"),
      svg("BarChart[{1, 2, 3}]")
    );
  }

  #[test]
  fn sector_chart_vector_time_series_keeps_the_pairs() {
    assert_eq!(
      svg(
        "SectorChart[TimeSeries[{{19, 16}, {9, 3}, {7, 2}, {17, 5}}, \
         {\"May 24, 1982\"}]]"
      ),
      svg("SectorChart[{{19, 16}, {9, 3}, {7, 2}, {17, 5}}]")
    );
  }

  #[test]
  fn sector_chart_association_plots_the_values() {
    assert_eq!(
      svg("SectorChart[<|\"a\" -> {2, 3}, \"b\" -> {4, 6}, \"c\" -> {5, 8}|>]"),
      svg("SectorChart[{{2, 3}, {4, 6}, {5, 8}}]")
    );
  }

  #[test]
  fn sector_chart_3d_time_series_and_association() {
    assert_eq!(
      svg(&format!("SectorChart3D[{TS_3}]")),
      svg("SectorChart3D[{{19, 16, 10}, {9, 3, 6}, {7, 6, 2}, {17, 12, 5}}]")
    );
    assert_eq!(
      svg("SectorChart3D[<|\"a\" -> {1, 3, 1}, \"b\" -> {2, 4, 3}|>]"),
      svg("SectorChart3D[{{1, 3, 1}, {2, 4, 3}}]")
    );
  }

  #[test]
  fn rectangle_chart_quantity_data_plots_the_magnitudes() {
    assert_eq!(
      interpret(
        "RectangleChart[{{Quantity[3, \"Seconds\"], Quantity[2, \"Meters\"]}, \
         {Quantity[1, \"Seconds\"], Quantity[4, \"Meters\"]}}]"
      )
      .unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      interpret(
        "RectangleChart[{{Quantity[3, \"Seconds\"], Quantity[2, \"Meters\"]}, \
         {Quantity[1, \"Seconds\"], Quantity[4, \"Meters\"]}}] === \
         RectangleChart[{{3, 2}, {1, 4}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Head[RectangleChart3D[{{Quantity[3, \"Seconds\"], \
         Quantity[5, \"Seconds\"], Quantity[2, \"Meters\"]}}, \
         BarSpacing -> None]]"
      )
      .unwrap(),
      "Graphics3D"
    );
  }

  #[test]
  fn rectangle_chart_time_series_and_association() {
    assert_eq!(
      interpret(
        "RectangleChart[TimeSeries[{{19, 16}, {9, 3}}, {\"May 24, 1982\"}]] \
         === RectangleChart[{{19, 16}, {9, 3}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "RectangleChart[<|\"a\" -> {1, 1}, \"b\" -> {1, 2}|>] === \
         RectangleChart[{{1, 1}, {1, 2}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "RectangleChart3D[<|\"a\" -> {1, 1, 1}, \"b\" -> {1, 3, 2}|>] === \
         RectangleChart3D[{{1, 1, 1}, {1, 3, 2}}]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn bubble_chart_time_series_plots_the_values() {
    assert_eq!(
      svg(
        "BubbleChart[TimeSeries[{{23, 12, 24}, {15, 3, 9}, {3, 23, 12}}, \
         {\"May 24, 1982\"}]]"
      ),
      svg("BubbleChart[{{23, 12, 24}, {15, 3, 9}, {3, 23, 12}}]")
    );
  }

  #[test]
  fn non_data_still_reports_ldata() {
    assert_eq!(interpret("BarChart3D[x]").unwrap(), "BarChart3D[x]");
    assert_eq!(interpret("SectorChart[x]").unwrap(), "SectorChart[x]");
  }
}

mod chart_labels_and_legends {
  use super::*;

  #[test]
  fn bar_chart_3d_keys_as_labels() {
    let s = svg(
      "BarChart3D[<|\"ka\" -> 4, \"kb\" -> 2, \"kc\" -> 6|>, \
       PlotLabels -> Automatic]",
    );
    assert_texts(&s, &["ka", "kb", "kc"]);
    assert_texts(
      &svg("BarChart3D[{1, 2}, ChartLabels -> {\"x1\", \"x2\"}]"),
      &["x1", "x2"],
    );
  }

  #[test]
  fn bar_chart_3d_keys_as_legends_with_plot_style() {
    let plain = svg("BarChart3D[{4, 2, 6}]");
    let s = svg(
      "BarChart3D[<|\"ka\" -> 4, \"kb\" -> 2, \"kc\" -> 6|>, \
       PlotStyle -> {Red, Green, Blue}, PlotLegends -> Automatic]",
    );
    assert_texts(&s, &["ka", "kb", "kc"]);
    // The frame widens for the legend; each legend swatch has its colour.
    assert!(view_width(&s) > view_width(&plain));
    assert!(fills(&s, "255,0,0") >= 1);
    assert!(fills(&s, "0,0,255") >= 1);
  }

  #[test]
  fn pie_chart_3d_keys_as_labels_and_legends() {
    assert_texts(
      &svg(
        "PieChart3D[<|\"ka\" -> 1, \"kb\" -> 2, \"kc\" -> 5|>, \
         PlotLabels -> Automatic]",
      ),
      &["ka", "kb", "kc"],
    );
    let s = svg(
      "PieChart3D[<|\"ka\" -> 1, \"kb\" -> 2|>, PlotStyle -> {Red, Blue}, \
       PlotLegends -> Automatic]",
    );
    assert_texts(&s, &["ka", "kb"]);
    assert_eq!(fills(&s, "255,0,0"), 1);
  }

  #[test]
  fn sector_chart_callout_and_placed_labels() {
    for spec in ["Callout[Automatic]", "Placed[Automatic, \"RadialCallout\"]"] {
      let s = svg(&format!(
        "SectorChart[<|\"ka\" -> {{2, 3}}, \"kb\" -> {{4, 6}}, \
         \"kc\" -> {{5, 8}}|>, PlotLabels -> {spec}]"
      ));
      assert_texts(&s, &["ka", "kb", "kc"]);
      // One leader line per callout.
      assert_eq!(s.matches("<line").count(), 3, "{spec}");
    }
    let s =
      svg("SectorChart[{{1, 1}, {1, 2}}, ChartLabels -> {\"la\", \"lb\"}]");
    assert_texts(&s, &["la", "lb"]);
    assert_eq!(s.matches("<line").count(), 0);
  }

  #[test]
  fn sector_chart_plot_style_and_legends() {
    let s = svg(
      "SectorChart[<|\"ka\" -> {2, 3}, \"kb\" -> {4, 6}|>, \
       PlotStyle -> {Red, Blue}, PlotLegends -> Automatic]",
    );
    assert_texts(&s, &["ka", "kb"]);
    // Sector and legend swatch each.
    assert_eq!(fills(&s, "255,0,0"), 2);
    assert_eq!(fills(&s, "0,0,255"), 2);
  }

  #[test]
  fn sector_chart_3d_keys_as_labels_and_legends() {
    assert_texts(
      &svg(
        "SectorChart3D[<|\"ka\" -> {1, 3, 1}, \"kb\" -> {2, 4, 3}|>, \
         PlotLabels -> Automatic]",
      ),
      &["ka", "kb"],
    );
    assert_texts(
      &svg(
        "SectorChart3D[<|\"ka\" -> {1, 3, 1}, \"kb\" -> {2, 4, 3}|>, \
         PlotLegends -> Automatic]",
      ),
      &["ka", "kb"],
    );
  }

  #[test]
  fn rectangle_chart_plot_option_names() {
    assert_eq!(
      interpret(
        "RectangleChart[<|\"a\" -> {1, 1}, \"b\" -> {1, 2}|>, \
         PlotLabels -> Automatic] === \
         RectangleChart[{{1, 1}, {1, 2}}, ChartLabels -> {\"a\", \"b\"}]"
      )
      .unwrap(),
      "True"
    );
    assert_texts(
      &svg(
        "RectangleChart[<|\"ka\" -> {1, 1}, \"kb\" -> {1, 2}|>, \
         PlotLegends -> Automatic, PlotStyle -> {Red, Blue}]",
      ),
      &["ka", "kb"],
    );
  }

  #[test]
  fn rectangle_chart_callout_labels_sit_above_the_bars() {
    // The label of the {1, 2} rectangle is anchored at its top.
    assert_eq!(
      interpret(
        "Cases[RectangleChart[<|\"a\" -> {1, 1}, \"b\" -> {1, 2}|>, \
         PlotLabels -> Callout[Automatic, Above]], \
         Text[\"b\", {_, y_}, ___] :> y, Infinity]"
      )
      .unwrap(),
      "{2.}"
    );
  }

  #[test]
  fn bubble_chart_keys_as_labels() {
    assert_texts(
      &svg(
        "BubbleChart[<|\"ka\" -> {3, 1, 5}, \"kb\" -> {2, 2, 4}|>, \
         PlotLabels -> Automatic]",
      ),
      &["ka", "kb"],
    );
  }
}

mod list_line_plot_3d {
  use super::*;

  #[test]
  fn quantity_data_plots_the_magnitudes() {
    let data = "{{0.1, 0.2, 0.3}, {0.4, 0.1, 0.9}, {0.7, 0.5, 0.2}}";
    assert_eq!(
      svg(&format!(
        "ListLinePlot3D[Quantity[{data}, \"Meters\"], AxesLabel -> Automatic]"
      )),
      svg(&format!("ListLinePlot3D[{data}]"))
    );
  }

  #[test]
  fn list_of_height_matrices_is_one_dataset_each() {
    let s = svg(
      "ListLinePlot3D[Table[Table[Sin[i j k/10.], {i, 4}, {j, 6}], {k, 2}]]",
    );
    // Two datasets → two curve colours, each drawn by four row curves.
    assert!(s.contains("stroke=\"rgb(94,129,181)\""));
    assert!(s.contains("stroke=\"rgb(224,147,44)\""));
    assert!(!s.contains("stroke=\"rgb(143,176,50)\""));
  }

  #[test]
  fn plot_legends_names_the_datasets() {
    assert_eq!(
      interpret(
        "Head[ListLinePlot3D[Table[Table[i j k, {i, 3}, {j, 4}], {k, 2}], \
         PlotLegends -> {\"Austin\", \"Denver\"}]]"
      )
      .unwrap(),
      "Legended"
    );
    assert_texts(
      &svg(
        "ListLinePlot3D[Table[Table[i j k, {i, 3}, {j, 4}], {k, 2}], \
         PlotLegends -> {\"Austin\", \"Denver\"}]",
      ),
      &["Austin", "Denver"],
    );
  }
}

mod ternary_list_plot {
  use super::*;

  const POINTS: &str = "{{1, 2, 3}, {3, 1, 1}, {2, 2, 5}}";

  #[test]
  fn labeled_data_labels_the_points() {
    let s = svg(&format!(
      "TernaryListPlot[{POINTS} -> {{\"pa\", \"pb\", \"pc\"}}]"
    ));
    assert_texts(&s, &["pa", "pb", "pc"]);
    assert_eq!(s.matches("<circle").count(), 3);
    let s = svg(&format!("TernaryListPlot[{POINTS} -> Range[3]]"));
    assert_texts(&s, &["1", "2", "3"]);
  }

  #[test]
  fn association_keys_as_dataset_labels() {
    let s = svg(&format!(
      "TernaryListPlot[<|\"One\" -> {POINTS}, \"Two\" -> {POINTS}|>, \
       PlotStyle -> PointSize[0.02], PlotLabels -> Automatic]"
    ));
    assert_texts(&s, &["One", "Two"]);
    assert_eq!(s.matches("<circle").count(), 6);
  }

  #[test]
  fn plot_legends_draw_a_point_legend() {
    let expr = format!(
      "TernaryListPlot[{{{POINTS}, {POINTS}}}, PlotLegends -> {{\"da\", \"db\"}}]"
    );
    assert_eq!(interpret(&format!("Head[{expr}]")).unwrap(), "Legended");
    assert_texts(&svg(&expr), &["da", "db"]);
  }

  #[test]
  fn quantity_coordinates_plot_as_magnitudes() {
    assert_eq!(
      svg(
        "TernaryListPlot[{{Quantity[1, \"Meters\"], Quantity[2, \"Meters\"], \
         Quantity[3, \"Meters\"]}}]"
      ),
      svg("TernaryListPlot[{{1, 2, 3}}]")
    );
  }
}

mod list_fit_plot {
  use super::*;

  #[test]
  fn labeled_data_labels_the_points_and_fits_the_values() {
    let s = svg(
      "ListFitPlot[{1, 1, 2, 3, 5, 8} -> {\"la\", \"lb\", \"lc\", \"ld\", \
       \"le\", \"lf\"}]",
    );
    assert_texts(&s, &["la", "lb", "lc", "ld", "le", "lf"]);
    assert_eq!(
      interpret("Length[ListFitPlot[{1, 1, 2, 3, 5, 8} -> Range[6]][[1]]] > 0")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn show_keeps_point_labels() {
    let s = svg("Show[ListPlot[{Labeled[1, \"qa\"], 2}], Plot[x, {x, 1, 2}]]");
    assert_texts(&s, &["qa"]);
  }
}

mod point_legend {
  use super::*;

  #[test]
  fn legended_point_legend_draws_its_labels() {
    let s = svg(
      "Legended[Graphics[Disk[]], PointLegend[{Red, Blue}, {\"pa\", \"pb\"}]]",
    );
    assert!(s.contains("pa") && s.contains("pb"));
    assert!(s.contains("<circle"));
  }
}
