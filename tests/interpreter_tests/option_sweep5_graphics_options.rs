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
    // `Thick` is 2 display pixels: 20 render units (the width of the
    // default curve, which both plots draw once).
    assert_eq!(count(&plain, "stroke-width", "20"), 1);
    assert!(count(&thick, "stroke-width", "20") > 1);
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

  /// The `(r, fill, opacity)` of every mesh dot, in render units.
  fn dots(s: &str) -> Vec<(u32, String, String)> {
    let attr = |c: &str, name: &str| {
      let key = format!(" {name}=\"");
      let start = c.find(&key)? + key.len();
      Some(c[start..start + c[start..].find('"')?].to_string())
    };
    s.split("<circle")
      .skip(1)
      .map(|c| {
        (
          attr(c, "r").unwrap().parse().unwrap(),
          attr(c, "fill").unwrap(),
          attr(c, "opacity").unwrap_or_default(),
        )
      })
      .collect()
  }

  /// The `(stroke, opacity)` of every curve piece.
  fn strokes(s: &str) -> Vec<(String, String)> {
    let attr = |c: &str, name: &str| {
      let key = format!(" {name}=\"");
      let start = c.find(&key)? + key.len();
      Some(c[start..start + c[start..].find('"')?].to_string())
    };
    s.split("<polyline")
      .skip(1)
      .map(|c| {
        (
          attr(c, "stroke").unwrap_or_default(),
          attr(c, "opacity").unwrap_or_default(),
        )
      })
      .collect()
  }

  fn count_fill(s: &str, fill: &str) -> usize {
    dots(s).iter().filter(|d| d.1 == fill).count()
  }

  #[test]
  fn explicit_mesh_values_mark_the_curve() {
    // WL samples just inside the domain, so 0 and 2 Pi are not marked.
    let s = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> {Range[0, 2 Pi, Pi/4]}, \
       MeshStyle -> PointSize[Medium]]",
    );
    let d = dots(&s);
    assert_eq!(d.len(), 7, "{d:?}");
    // `PointSize[Medium]` is a 4.5 pt dot (radius 2.25 pt).
    assert!(d.iter().all(|c| c.0 == 23), "{d:?}");
    // A list plot's first data point is a mesh point when it takes a value.
    let l = svg(
      "ListPlot[Table[{x, Sin[x]}, {x, 0, 2 Pi, 0.1}], \
       Mesh -> {Range[0, 2 Pi, Pi/4]}, MeshStyle -> PointSize[Medium], \
       Joined -> True]",
    );
    assert_eq!(circles(&l), 8);
    assert_eq!(
      circles(&svg(
        "Plot[Sin[Sqrt[2] x] + Sin[x], {x, 0, 20}, \
         MeshFunctions -> {Function[{x, y}, y]}, Mesh -> {{0}}, \
         MeshStyle -> Directive[PointSize[Medium], Red]]"
      )),
      8
    );
  }

  #[test]
  fn mesh_values_carry_their_own_styles() {
    let s = svg(
      "ListPlot[Table[{x, Sin[x]}, {x, 0, 2 Pi, 2 Pi/20}], Mesh -> \
       {Table[{x, Directive[Hue[x/(2 Pi)], PointSize[Medium]]}, \
       {x, 0., 2 Pi, 2 Pi/8}]}, Joined -> True]",
    );
    let d = dots(&s);
    assert_eq!(d.len(), 9, "{d:?}");
    // Hue[0] and Hue[1] at the two ends, Hue[0.5] in the middle.
    assert_eq!(count_fill(&s, "#FF0000"), 2, "{d:?}");
    assert_eq!(count_fill(&s, "#00FFFF"), 1, "{d:?}");
    assert!(d.iter().all(|c| c.0 == 23), "{d:?}");
  }

  #[test]
  fn several_mesh_functions_take_their_own_counts_and_styles() {
    let s = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 10, MeshStyle -> {Red, Blue}, \
       MeshFunctions -> {#1 &, #2 &}]",
    );
    assert_eq!(count_fill(&s, "#FF0000"), 10);
    assert_eq!(count_fill(&s, "#0000FF"), 20);
    let l = svg(
      "ListPlot[Table[Binomial[15, k], {k, 0, 15}], Mesh -> {5, 10}, \
       MeshFunctions -> {#1 &, #2 &}, \
       MeshStyle -> {Directive[PointSize[Medium], Red], Blue}, \
       Joined -> True]",
    );
    let d = dots(&l);
    assert_eq!(
      d.iter().filter(|c| c.1 == "#FF0000" && c.0 == 23).count(),
      5
    );
    assert_eq!(count_fill(&l, "#0000FF"), 20);
  }

  #[test]
  fn mesh_style_point_size_sets_the_dot_radius() {
    let s = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 10, \
       MeshStyle -> Directive[PointSize[Large], Red]]",
    );
    let d = dots(&s);
    assert_eq!(d.len(), 10);
    assert!(d.iter().all(|c| c.0 == 35 && c.1 == "#FF0000"), "{d:?}");
    // Without `MeshStyle`, a curve's mesh dots are 3 pt across …
    assert!(
      dots(&svg("Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 3]"))
        .iter()
        .all(|c| c.0 == 15)
    );
    // … and a ListLinePlot's take its automatic PointSize[7/360].
    assert!(
      dots(&svg("ListLinePlot[Range[8], Mesh -> All]"))
        .iter()
        .all(|c| c.0 == 35)
    );
    // `MeshStyle -> Opacity[…]` alone is a style too.
    let o =
      svg("Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 2, MeshStyle -> Opacity[0.3]]");
    assert!(dots(&o).iter().all(|c| c.2 == "0.3"), "{:?}", dots(&o));
  }

  #[test]
  fn y_mesh_function_marks_equally_spaced_heights() {
    // Heights -1/2, 0, 1/2; the zero at x = 0 is the curve's end.
    assert_eq!(
      circles(&svg(
        "Plot[Sin[x], {x, 0, 2 Pi}, MeshFunctions -> {Function[{x, y}, y]}, \
         Mesh -> 3]"
      )),
      5
    );
    assert_eq!(
      circles(&svg(
        "ListPlot[Table[Binomial[15, k], {k, 0, 15}], \
         MeshFunctions -> {Function[{x, y}, y]}, Mesh -> 9, Joined -> True]"
      )),
      18
    );
  }

  #[test]
  fn mesh_values_span_every_curve_together() {
    // One value at x = 150.5, which only the longer curve reaches.
    assert_eq!(
      circles(&svg("ListLinePlot[{Range[8], Range[300]}, Mesh -> 1]")),
      1
    );
  }

  #[test]
  fn plot_mesh_full_marks_the_initial_grid() {
    assert_eq!(
      circles(&svg("Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> Full]")),
      49
    );
  }

  #[test]
  fn mesh_shading_alternates_the_curve_styles() {
    let s = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 10, MeshFunctions -> {#1 &}, \
       MeshShading -> {Red, Blue}]",
    );
    let st: Vec<String> = strokes(&s).into_iter().map(|p| p.0).collect();
    assert_eq!(st.len(), 11, "{st:?}");
    for (i, c) in st.iter().enumerate() {
      assert_eq!(c, if i % 2 == 0 { "#FF0000" } else { "#0000FF" });
    }
    // The mesh points keep the curve's colour.
    assert_eq!(count_fill(&s, "#3D99CC"), 10);
    // `None` leaves gaps; a three-entry list cycles.
    let gaps =
      svg("Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 10, MeshShading -> {Red, None}]");
    let st: Vec<String> = strokes(&gaps).into_iter().map(|p| p.0).collect();
    assert_eq!(st, vec!["#FF0000"; 6]);
    let three = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 10, \
       MeshShading -> {Red, None, Blue}]",
    );
    let st: Vec<String> = strokes(&three).into_iter().map(|p| p.0).collect();
    assert_eq!(
      st,
      [
        "#FF0000", "#0000FF", "#FF0000", "#0000FF", "#FF0000", "#0000FF",
        "#FF0000"
      ]
    );
  }

  #[test]
  fn mesh_shading_regions_follow_the_mesh_function_value() {
    // Regions are numbered by how many mesh values lie below the
    // function: y < -1/2 is Red, then Blue, Green, and y > 1/2 Orange.
    let s = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> 3, MeshFunctions -> {#2 &}, \
       MeshShading -> {Red, Blue, Green, Orange}]",
    );
    let st: Vec<String> = strokes(&s).into_iter().map(|p| p.0).collect();
    assert_eq!(
      st,
      [
        "#00FF00", "#FF8000", "#00FF00", "#0000FF", "#FF0000", "#0000FF"
      ]
    );
    // Two mesh functions need a rank-2 array; a flat list shades nothing.
    let flat = svg(
      "Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> {2, 1}, \
       MeshFunctions -> {#1 &, #2 &}, MeshShading -> {Red, Blue, Green}]",
    );
    assert!(strokes(&flat).iter().all(|p| p.0 == "#3D99CC"));
  }

  #[test]
  fn mesh_shading_keeps_the_plot_style() {
    // `Automatic` stretches draw in the PlotStyle colour.
    let auto = svg(
      "Plot[Sin[x], {x, 0, 10}, Mesh -> 10, \
       PlotStyle -> Directive[Thick, Yellow], MeshFunctions -> {#1 &}, \
       MeshShading -> {Red, Automatic}]",
    );
    let st: Vec<String> = strokes(&auto).into_iter().map(|p| p.0).collect();
    assert_eq!(st.iter().filter(|c| *c == "#FF0000").count(), 6);
    assert_eq!(st.iter().filter(|c| *c == "#FFFF00").count(), 5);
    // The PlotStyle opacity stays on the shaded stretches and the dots.
    let op = svg(
      "ListLinePlot[Table[Sin[x], {x, 0, 2 Pi, 0.1}], \
       PlotStyle -> Directive[Opacity[0.5], Thick], Mesh -> 10, \
       MeshFunctions -> {#1 &}, MeshShading -> {Red, Blue}]",
    );
    let st = strokes(&op);
    assert_eq!(st.len(), 11);
    assert!(st.iter().all(|p| p.1 == "0.5"), "{st:?}");
    assert!(dots(&op).iter().all(|d| d.2 == "0.5"));
    // `ColorFunction` colours the `Automatic` stretches and the dots.
    let cf = svg(
      "Plot[Sin[x], {x, 0, 10}, Mesh -> 10, MeshFunctions -> {#1 &}, \
       MeshShading -> {StandardGray, Automatic}, \
       ColorFunction -> Function[{x, y}, Hue[x]]]",
    );
    let fills: std::collections::HashSet<String> =
      dots(&cf).into_iter().map(|d| d.1).collect();
    assert_eq!(fills.len(), 10, "{fills:?}");
  }

  #[test]
  fn list_plot_mesh_shading_defaults_to_x() {
    // Without MeshFunctions, a joined ListPlot shades between x values.
    let s = svg(
      "ListPlot[Table[Binomial[15, k], {k, 0, 15}], Mesh -> 10, \
       MeshShading -> {Red, Blue}, Joined -> True]",
    );
    let st: Vec<String> = strokes(&s).into_iter().map(|p| p.0).collect();
    assert_eq!(st.len(), 11, "{st:?}");
    assert_eq!(st.iter().filter(|c| *c == "#0000FF").count(), 5);
    // An unjoined ListPlot has no mesh: just its 16 points.
    assert_eq!(
      circles(&svg(
        "ListPlot[Table[Binomial[15, k], {k, 0, 15}], Mesh -> 10, \
         MeshShading -> {Red, Blue}]"
      )),
      16
    );
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
        .map(|p| p.split("/>").next().unwrap_or(p))
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
        .find(|r| r.contains("#FFD540") || r.contains("rgb(255,213,64)"))
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
