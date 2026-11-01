// Styling options of the core plot and layout functions that the
// reference-page parameter sweep (round 5) found accepted but without any
// visible effect. Graphics are checked by their structure — the strokes,
// fills and primitives an option adds — not by their full SVG text.

use super::*;

/// The SVG `ExportString` draws for `expr`.
fn svg(expr: &str) -> String {
  let out = interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap();
  assert!(out.starts_with("<svg"), "{expr} gave no picture: {out}");
  out
}

/// How many elements carry `attr="value"`.
fn count(svg: &str, attr: &str, value: &str) -> usize {
  svg.matches(&format!("{attr}=\"{value}\"")).count()
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

const RED_RGB: &str = "rgb(255,0,0)";
const RED_HEX: &str = "#FF0000";

mod axes_and_ticks_style {
  use super::*;

  #[test]
  fn plot_axes_style_colours_axes_ticks_and_labels() {
    let plain = svg("Plot[Sin[x], {x, 0, 6}]");
    assert_eq!(count(&plain, "stroke", RED_RGB), 0);
    let s = svg("Plot[Sin[x], {x, 0, 6}, AxesStyle -> Red]");
    // Both axis lines and every tick mark.
    assert!(count(&s, "stroke", RED_RGB) > 10, "{s}");
    // The tick labels follow the axes.
    assert!(count(&s, "fill", RED_RGB) >= 5);
  }

  #[test]
  fn axes_style_pair_styles_each_axis() {
    let s = svg("Plot[Sin[x], {x, 0, 6}, AxesStyle -> {Red, Blue}]");
    assert!(count(&s, "stroke", RED_RGB) > 0);
    assert!(count(&s, "stroke", "rgb(0,0,255)") > 0);
  }

  #[test]
  fn axes_style_thickness_widens_the_axis_lines() {
    let plain = svg("Plot[Sin[x], {x, 0, 6}]");
    let thick = svg("Plot[Sin[x], {x, 0, 6}, AxesStyle -> Thick]");
    // `Thick` is 2 display pixels: 20 render units.
    assert_eq!(count(&plain, "stroke-width", "20"), 0);
    assert!(count(&thick, "stroke-width", "20") > 0);
  }

  #[test]
  fn ticks_style_leaves_the_axis_lines_alone() {
    let s = svg("Plot[Sin[x], {x, 0, 6}, TicksStyle -> Red]");
    let red_lines = count(&s, "stroke", RED_RGB);
    assert!(red_lines > 10);
    // Ticks and labels are red, the two axis lines are not.
    let all_lines = s.matches("<line").count();
    assert!(red_lines < all_lines);
    assert!(count(&s, "fill", RED_RGB) >= 5);
  }

  #[test]
  fn list_plots_and_polar_plot_share_the_axes_style() {
    for expr in [
      "ListPlot[Range[10], AxesStyle -> Red]",
      "ListLinePlot[Range[10], TicksStyle -> Red]",
      "PolarPlot[1 + Cos[t], {t, 0, 6}, AxesStyle -> Red]",
      "LogPlot[Exp[x], {x, 0, 5}, AxesStyle -> Red]",
      "DiscretePlot[n, {n, 1, 5}, AxesStyle -> Red]",
    ] {
      assert!(count(&svg(expr), "stroke", RED_RGB) > 0, "{expr}");
    }
  }

  #[test]
  fn graphics_ticks_style_colours_tick_marks_and_labels() {
    let plain = svg("Graphics[{Disk[]}, Axes -> True]");
    assert_eq!(count(&plain, "stroke", RED_RGB), 0);
    let s = svg("Graphics[{Disk[]}, Axes -> True, TicksStyle -> Red]");
    assert!(count(&s, "stroke", RED_RGB) > 4);
    assert!(count(&s, "fill", RED_RGB) > 2);
  }

  #[test]
  fn graphics_axes_origin_moves_the_axes() {
    let s = svg("Graphics[{Disk[]}, Axes -> True, AxesOrigin -> {0.5, 0.5}]");
    let plain = svg("Graphics[{Disk[]}, Axes -> True]");
    assert_ne!(s, plain);
    // The horizontal axis spans the width at the height of y = 0.5: a
    // quarter of the way down from the top of the [-1, 1] range.
    let first_line = s.split("<line").nth(1).unwrap();
    let plain_line = plain.split("<line").nth(1).unwrap();
    assert_ne!(first_line, plain_line);
  }
}

mod frame_style {
  use super::*;

  #[test]
  fn plot_frame_style_colours_and_thickens_the_frame() {
    let s = svg(
      "Plot[Sin[x], {x, 0, 6}, Frame -> True, \
       FrameStyle -> Directive[Red, Thick]]",
    );
    assert!(s.contains("stroke=\"#FF0000\" stroke-width=\"20\""), "{s}");
    // Frame tick labels take the frame's colour.
    assert!(count(&s, "fill", RED_HEX) > 0);
  }

  #[test]
  fn list_plot_frame_style() {
    let plain = svg("ListPlot[{1, 5, 2, 8}, Frame -> True]");
    assert_eq!(count(&plain, "stroke", RED_HEX), 0);
    let s = svg("ListPlot[{1, 5, 2, 8}, Frame -> True, FrameStyle -> Red]");
    assert!(count(&s, "stroke", RED_HEX) > 0);
  }

  #[test]
  fn graphics_frame_style_and_frame_ticks_style() {
    let s = svg("Graphics[{Disk[]}, Frame -> True, FrameStyle -> Red]");
    assert!(s.contains("fill=\"none\" stroke=\"rgb(255,0,0)\""), "{s}");
    let t = svg("Graphics[{Disk[]}, Frame -> True, FrameTicksStyle -> Red]");
    // The border keeps its colour; the tick marks and labels turn red.
    assert!(!t.contains("fill=\"none\" stroke=\"rgb(255,0,0)\""));
    assert!(count(&t, "stroke", RED_RGB) > 4);
    assert!(count(&t, "fill", RED_RGB) > 2);
  }
}

mod label_and_base_style {
  use super::*;

  #[test]
  fn graphics_label_style_colours_the_plot_label() {
    let s = svg("Graphics[{Disk[]}, PlotLabel -> \"hi\", LabelStyle -> Red]");
    assert!(s.contains(&format!("fill=\"{RED_RGB}\"")), "{s}");
    assert!(texts(&s).iter().any(|t| t.contains("hi")));
  }

  #[test]
  fn graphics_base_style_is_the_default_style_of_the_primitives() {
    let plain = svg("Graphics[{Line[{{0, 0}, {1, 1}}]}]");
    let s = svg("Graphics[{Line[{{0, 0}, {1, 1}}]}, BaseStyle -> Red]");
    assert_ne!(plain, s);
    assert!(s.contains(RED_RGB) || s.contains("rgb(100%,0%,0%)"), "{s}");
    // A directive in the drawing still overrides the base style.
    let blue =
      svg("Graphics[{Blue, Line[{{0, 0}, {1, 1}}]}, BaseStyle -> Red]");
    assert_eq!(
      blue,
      svg("Graphics[{Blue, Line[{{0, 0}, {1, 1}}]}]"),
      "explicit colour must win"
    );
  }
}

mod mesh {
  use super::*;

  /// The number of mesh dots drawn (plotters writes them as circles).
  fn circles(s: &str) -> usize {
    s.matches("<circle").count()
  }

  #[test]
  fn plot_mesh_count_draws_that_many_points() {
    assert_eq!(circles(&svg("Plot[Sin[x], {x, 0, 6}]")), 0);
    assert_eq!(circles(&svg("Plot[Sin[x], {x, 0, 6}, Mesh -> 5]")), 5);
    assert!(circles(&svg("Plot[Sin[x], {x, 0, 6}, Mesh -> All]")) > 20);
  }

  #[test]
  fn mesh_style_colours_the_mesh_points() {
    let s = svg("Plot[Sin[x], {x, 0, 6}, Mesh -> 5, MeshStyle -> Red]");
    let red_circles = s
      .split("<circle")
      .skip(1)
      .filter(|c| c.contains(RED_HEX))
      .count();
    assert_eq!(red_circles, 5, "{s}");
    let l = svg("ListLinePlot[Range[10], Mesh -> All, MeshStyle -> Red]");
    assert!(l.split("<circle").skip(1).any(|c| c.contains(RED_HEX)));
  }

  #[test]
  fn polar_and_log_plots_take_a_mesh() {
    assert_eq!(
      circles(&svg("PolarPlot[1 + Cos[t], {t, 0, 6}, Mesh -> 5]")),
      5
    );
    assert_eq!(circles(&svg("LogPlot[Exp[x], {x, 0, 5}, Mesh -> 4]")), 4);
  }
}

mod labels_and_legends {
  use super::*;

  #[test]
  fn list_plot_labels_name_the_series() {
    let s = svg(
      "ListPlot[{{1, 5, 2, 8}, {2, 3, 4, 1}}, PlotLabels -> {\"aa\", \"bb\"}]",
    );
    let t = texts(&s);
    assert!(t.iter().any(|x| x == "aa") && t.iter().any(|x| x == "bb"));
  }

  #[test]
  fn parametric_and_polar_plots_draw_legends_and_labels() {
    for expr in [
      "ParametricPlot[{Sin[t], Cos[t]}, {t, 0, 6}, PlotLegends -> {\"aa\"}]",
      "ParametricPlot[{Sin[t], Cos[t]}, {t, 0, 6}, PlotLabels -> {\"aa\"}]",
      "PolarPlot[1 + Cos[t], {t, 0, 6}, PlotLegends -> {\"aa\"}]",
    ] {
      assert!(texts(&svg(expr)).iter().any(|x| x == "aa"), "{expr}");
    }
  }
}

mod scaling_functions {
  use super::*;

  #[test]
  fn log_scaling_is_the_log_plot() {
    assert_eq!(
      svg("Plot[Exp[x], {x, 0, 6}, ScalingFunctions -> \"Log\"]"),
      svg("LogPlot[Exp[x], {x, 0, 6}]")
    );
    assert_eq!(
      svg("Plot[x, {x, 1, 100}, ScalingFunctions -> {\"Log\", \"Log\"}]"),
      svg("LogLogPlot[x, {x, 1, 100}]")
    );
    assert_eq!(
      svg("Plot[x, {x, 1, 100}, ScalingFunctions -> {\"Log\", None}]"),
      svg("LogLinearPlot[x, {x, 1, 100}]")
    );
  }

  #[test]
  fn list_plots_scale_through_the_list_log_plots() {
    assert_eq!(
      svg("ListPlot[{1, 10, 100}, ScalingFunctions -> \"Log\"]"),
      svg("ListLogPlot[{1, 10, 100}]")
    );
    assert_eq!(
      svg("ListLinePlot[{1, 10, 100}, ScalingFunctions -> \"Log\"]"),
      svg("ListLogPlot[{1, 10, 100}, Joined -> True]")
    );
  }

  #[test]
  fn no_scaling_keeps_the_plain_plot() {
    assert_eq!(
      svg("Plot[Sin[x], {x, 0, 6}, ScalingFunctions -> None]"),
      svg("Plot[Sin[x], {x, 0, 6}]")
    );
  }
}

mod charts {
  use super::*;

  #[test]
  fn bar_chart_color_function_colours_by_value() {
    let s = svg("BarChart[{1, 2, 3}, ColorFunction -> (RGBColor[#, 0, 0] &)]");
    // The values rescale to 0, 1/2 and 1.
    for c in ["#000000", "#800000", "#FF0000"] {
      assert!(count(&s, "fill", c) >= 1, "missing {c}");
    }
  }

  #[test]
  fn pie_chart_and_histogram_color_function() {
    let p = svg("PieChart[{1, 2, 3}, ColorFunction -> (RGBColor[#, 0, 0] &)]");
    assert!(count(&p, "fill", "rgb(128,0,0)") == 1, "{p}");
    let h = svg(
      "Histogram[{1, 2, 2, 3, 3, 3}, ColorFunction -> (RGBColor[0, 0, #] &)]",
    );
    assert!(count(&h, "fill", "#0000FF") >= 1, "{h}");
  }

  #[test]
  fn color_function_scaling_false_passes_raw_values() {
    let s = svg(
      "BarChart[{0.25, 0.5}, ColorFunction -> (RGBColor[#, 0, 0] &), \
       ColorFunctionScaling -> False]",
    );
    assert!(count(&s, "fill", "#400000") >= 1, "{s}");
    assert!(count(&s, "fill", "#800000") >= 1);
  }

  #[test]
  fn bar_chart_axes_and_frame() {
    let plain = svg("BarChart[{1, 2, 3}]");
    let red = svg("BarChart[{1, 2, 3}, AxesStyle -> Red]");
    assert_eq!(count(&plain, "stroke", RED_HEX), 0);
    assert!(count(&red, "stroke", RED_HEX) > 0);
    let hidden = svg("BarChart[{1, 2, 3}, Axes -> False]");
    assert_ne!(hidden, plain);
    let framed = svg("BarChart[{1, 2, 3}, Frame -> True, FrameStyle -> Red]");
    // A closed path round the plotting area.
    assert!(
      framed
        .split("<polyline")
        .skip(1)
        .any(|p| p.contains(RED_HEX) && p.matches(',').count() == 5),
      "{framed}"
    );
  }

  #[test]
  fn bar_spacing_widens_the_bars() {
    let width = |s: &str| -> f64 {
      // Width of the first bar (a filled rect, not the background).
      let rect = s
        .split("<rect")
        .skip(1)
        .find(|r| r.contains("#5E81B5") || r.contains("rgb(94,129,181)"))
        .unwrap();
      let w = rect.split("width=\"").nth(1).unwrap();
      w[..w.find('"').unwrap()].parse().unwrap()
    };
    let tight = svg("BarChart[{1, 2, 3}, BarSpacing -> 0]");
    let wide = svg("BarChart[{1, 2, 3}, BarSpacing -> 1]");
    assert!(width(&tight) > width(&wide));
  }

  #[test]
  fn bar_chart_aspect_ratio_sets_the_height() {
    let height = |s: &str| -> f64 {
      let h = s.split("height=\"").nth(1).unwrap();
      h[..h.find('"').unwrap()].parse().unwrap()
    };
    let square = svg("BarChart[{1, 2, 3}, AspectRatio -> 1]");
    let flat = svg("BarChart[{1, 2, 3}, AspectRatio -> 1/4]");
    assert!(height(&square) > 2.0 * height(&flat));
  }
}

mod layout_boxes {
  use super::*;

  #[test]
  fn framed_frame_style() {
    let s = svg("Framed[1, FrameStyle -> Directive[Red, Thick]]");
    assert!(
      s.contains(&format!("stroke=\"{RED_RGB}\" stroke-width=\"2\"")),
      "{s}"
    );
  }

  #[test]
  fn grid_frame_style_colours_the_frame() {
    let s = svg("Grid[{{1, 2}, {3, 4}}, Frame -> All, FrameStyle -> Red]");
    assert_eq!(count(&s, "stroke", RED_RGB), 6);
  }

  #[test]
  fn column_and_row_background_and_frame() {
    let c = svg("Column[{1, 2}, Background -> Yellow]");
    assert_eq!(count(&c, "fill", "rgb(255,255,0)"), 1);
    let r = svg("Row[{1, 2}, Background -> Yellow]");
    assert_eq!(count(&r, "fill", "rgb(255,255,0)"), 1);
    let f = svg("Column[{1, 2}, Frame -> True, FrameStyle -> Red]");
    assert!(f.contains(&format!("fill=\"none\" stroke=\"{RED_RGB}\"")));
    // Without the options the box is bare.
    assert!(!svg("Column[{1, 2}]").contains("<rect"));
  }
}
