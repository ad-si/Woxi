//! Plot styling and labeling options checked against wolframscript (WL 15):
//! plot themes, legend placement and titles, `ImageSize -> UpTo[n]`,
//! `BaselinePosition -> Axis`, insets of plots, axis and grid styles.
use super::*;

fn svg_of(code: &str) -> String {
  interpret(&format!("ExportString[{code}, \"SVG\"]")).unwrap()
}

/// The value of attribute `name` on the first `<svg>` element.
fn root_attr(svg: &str, name: &str) -> f64 {
  let head = &svg[..svg.find('>').unwrap()];
  let start = head.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
  head[start..start + head[start..].find('"').unwrap()]
    .trim_end_matches("pt")
    .parse()
    .unwrap()
}

/// The `(y, height)` of every `<svg x=… y=…>` nested in `svg`.
fn nested_y_h(svg: &str) -> Vec<(f64, f64)> {
  svg
    .match_indices("<svg x=\"")
    .map(|(i, _)| {
      let tag = &svg[i..i + svg[i..].find('>').unwrap()];
      let attr = |name: &str| -> f64 {
        let start = tag.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
        tag[start..start + tag[start..].find('"').unwrap()]
          .parse()
          .unwrap()
      };
      (attr("y"), attr("height"))
    })
    .collect()
}

mod plot_theme {
  use super::*;

  /// `"Monochrome"`: black curves told apart by WL's dash patterns, ended
  /// square, as `Charting`ResolvePlotTheme["Monochrome", Plot]` lists them.
  #[test]
  fn monochrome_dashes_black_curves() {
    let svg = svg_of(
      r#"Plot[{Sin[x], Cos[x], Sin[2 x]}, {x, 0, 2 Pi}, PlotTheme -> "Monochrome"]"#,
    );
    assert!(svg.contains("stroke-dasharray=\"60.0,20.0\""), "{svg}");
    assert!(svg.contains("stroke-dasharray=\"20.0,20.0\""), "{svg}");
    assert!(!svg.contains("stroke-linecap=\"round\""), "{svg}");
    assert!(!svg.contains("#3D99CC"), "no default palette color");
    assert!(!svg.contains("rgb(61,153,204)"), "no default palette color");
  }

  /// `"Dark"` draws in the Scientific palette at 2 points.
  #[test]
  fn dark_uses_its_palette() {
    let svg =
      svg_of(r#"Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotTheme -> "Dark"]"#);
    assert!(svg.contains("#E65C0E"), "orange-red first curve: {svg}");
    assert!(svg.contains("#5D6DC1"), "slate blue second curve: {svg}");
  }

  /// `"Detailed"` legends every curve by its expression (`PlotLegends ->
  /// AllExpressions`), a lone one too.
  #[test]
  fn detailed_legends_its_curves() {
    let svg = super::super::graphics::without_italics(&svg_of(
      r#"Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotTheme -> "Detailed"]"#,
    ));
    assert!(svg.contains(">sin(x)</text>"), "{svg}");
    assert!(svg.contains(">cos(x)</text>"), "{svg}");
    let single = super::super::graphics::without_italics(&svg_of(
      r#"Plot[Sin[x], {x, 0, 2 Pi}, PlotTheme -> "Detailed"]"#,
    ));
    assert!(single.contains(">sin(x)</text>"), "{single}");
  }
}

mod legends {
  use super::*;

  /// A lone curve gets a legend only from explicit labels or
  /// `AllExpressions`; `"Expressions"`, `Automatic` and `"Placeholder"`
  /// leave it unlegended (`Head` is `Graphics` in WL).
  #[test]
  fn a_lone_curve_is_legended_only_explicitly() {
    assert_eq!(
      interpret(
        r#"Table[StringCount[ExportString[Plot[Sin[x], {x, 0, 1}, PlotLegends -> l], "SVG"], "font-size=\"140\""], {l, {"Expressions", AllExpressions, Automatic, "Placeholder", {"a"}}}]"#
      )
      .unwrap(),
      "{0, 1, 0, 0, 1}"
    );
  }

  /// `LineLegend[…, LegendLabel -> title]` titles the legend above its
  /// entries.
  #[test]
  fn line_legend_label_is_drawn() {
    let svg = svg_of(
      r#"Plot[{x, x^2}, {x, 0, 1}, PlotLegends -> LineLegend[{"linear", "square"}, LegendLabel -> "power"]]"#,
    );
    let y_of = |label: &str| -> f64 {
      let end = svg.find(&format!(">{label}</text>")).unwrap();
      let tag = &svg[svg[..end].rfind("<text ").unwrap()..end];
      let start = tag.find(" y=\"").unwrap() + 4;
      tag[start..start + tag[start..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    assert!(y_of("power") < y_of("linear"), "the title sits on top");
  }

  /// `Placed[…, {Right, Top}]` puts the legend inside the image, in its
  /// top right corner, rather than beside the plot.
  #[test]
  fn placed_right_top_is_inside_the_corner() {
    let plain = svg_of("Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}]");
    let svg = svg_of(
      r#"Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotLegends -> Placed[{"sin", "cos"}, {Right, Top}]]"#,
    );
    assert_eq!(root_attr(&svg, "width"), root_attr(&plain, "width"));
    let end = svg.find(">sin</text>").unwrap();
    let tag = &svg[svg[..end].rfind("<text ").unwrap()..end];
    let y: f64 = tag
      .split(" y=\"")
      .nth(1)
      .and_then(|t| t.split('"').next())
      .and_then(|v| v.parse().ok())
      .unwrap();
    let height = root_attr(&svg, "height") * 10.0;
    assert!(y < height * 0.2, "near the top: {y} of {height}");
  }
}

mod image_size {
  use super::*;

  /// `ImageSize -> UpTo[n]` caps both sides: a wide plot is `n` wide, a
  /// tall one `n` high (WL: 353 × 200 for the pair, 125 × 200 the tall one).
  #[test]
  fn up_to_caps_width_and_height() {
    let wide = svg_of("Plot[Sin[x], {x, 0, 10}, ImageSize -> UpTo[200]]");
    assert_eq!(root_attr(&wide, "width"), 200.0);
    assert!(root_attr(&wide, "height") < 200.0);
    let tall = svg_of(
      "Plot[Sin[x], {x, 0, 10}, AspectRatio -> 2, ImageSize -> UpTo[200]]",
    );
    assert_eq!(root_attr(&tall, "height"), 200.0);
    assert!(root_attr(&tall, "width") < 200.0);
  }

  /// `{UpTo[w], h}` keeps the height and narrows only as far as the
  /// aspect ratio asks; `{w, UpTo[h]}` likewise for the height.
  #[test]
  fn up_to_in_one_slot() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 10}, AspectRatio -> 2, ImageSize -> {UpTo[300], 100}]",
    );
    assert_eq!(root_attr(&svg, "height"), 100.0);
    assert!(root_attr(&svg, "width") < 100.0);
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 10}, AspectRatio -> 2, ImageSize -> {200, UpTo[100]}]",
    );
    assert_eq!(root_attr(&svg, "width"), 200.0);
    assert_eq!(root_attr(&svg, "height"), 100.0);
  }
}

/// `BaselinePosition -> Axis` lines the pictures of a list up on their x
/// axes: an axis along the bottom of one and the top of the other puts
/// the second picture lower.
#[test]
fn baseline_position_axis_aligns_a_list() {
  let svg = svg_of(
    "{Plot[x, {x, 0, 1}, BaselinePosition -> Axis], \
     Plot[x, {x, -1, 0}, BaselinePosition -> Axis]}",
  );
  let cells = nested_y_h(&svg);
  assert_eq!(cells.len(), 2, "{svg}");
  let ((y1, h1), (y2, _)) = (cells[0], cells[1]);
  assert!(y2 - y1 > h1 / 2.0, "{cells:?}");
  // Without the option both are centered on the line.
  let svg = svg_of("{Plot[x, {x, 0, 1}], Plot[x, {x, -1, 0}]}");
  let cells = nested_y_h(&svg);
  assert!((cells[0].0 - cells[1].0).abs() < 2.0, "{cells:?}");
}

mod insets {
  use super::*;

  /// An inset plot is transparent and laid out at its inset size, so its
  /// tick labels keep their point size (WL draws the `LightYellow` panel
  /// through it).
  #[test]
  fn inset_plot_is_transparent_and_relaid_out() {
    let svg = svg_of(
      "Graphics[{LightYellow, Rectangle[{0, 0}, {10, 6}], \
       Inset[Plot[x^2, {x, -1, 1}], {5, 3}, Center, 6]}]",
    );
    assert!(!svg.contains("fill=\"#FFFFFF\""), "no white plate: {svg}");
    let nested = &svg[svg.find("<svg x=\"").unwrap()..];
    let view_box = nested
      .split("viewBox=\"")
      .nth(1)
      .and_then(|v| v.split('"').next())
      .unwrap();
    let width: f64 =
      view_box.split_whitespace().nth(2).unwrap().parse().unwrap();
    assert!(width < 3000.0, "re-rendered below 300 points: {view_box}");
  }

  /// An `Epilog` inset of a plot hides nothing behind it, and its size
  /// argument sets its width.
  #[test]
  fn epilog_inset_plot_is_transparent() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 10}, Epilog -> Inset[Plot[Sin[x], {x, 0, 1}, \
       Frame -> True], {7, 0.6}, Center, 3]]",
    );
    assert_eq!(svg.matches("fill=\"#FFFFFF\"").count(), 1, "{svg}");
  }
}

/// `Disk[c, Offset[r]]` in an `Epilog` has a radius of `r` printer's
/// points.
#[test]
fn epilog_disk_with_offset_radius() {
  let svg = svg_of(
    "Plot[Floor[x], {x, -2, 3}, Epilog -> {Table[Disk[{i, i}, Offset[2.]], {i, -2, 2}], \
     Table[{EdgeForm[StandardGray], White, Disk[{i + 1, i}, Offset[2]]}, {i, -2, 2}]}]",
  );
  assert_eq!(svg.matches("rx=\"20.0\" ry=\"20.0\"").count(), 10, "{svg}");
}

mod axes {
  use super::*;

  /// The axis labels take their axis's `AxesStyle` color, the tick marks
  /// its thickness, and `TicksStyle`'s bare number sizes the tick labels.
  #[test]
  fn axes_style_reaches_labels_and_ticks() {
    let svg = svg_of(
      r#"Plot[Sin[x], {x, 0, 2 Pi}, AxesStyle -> Directive[Blue, Thick], TicksStyle -> Directive[Red, 14], AxesLabel -> {"x", "sin(x)"}]"#,
    );
    assert!(svg.contains("fill=\"rgb(0,0,255)\">x</text>"), "{svg}");
    assert!(svg.contains("fill=\"rgb(0,0,255)\">sin(x)</text>"), "{svg}");
    assert!(
      svg.contains("font-size=\"140\" fill=\"rgb(255,0,0)\">1</text>"),
      "14 pt red tick labels: {svg}"
    );
    assert!(
      svg.contains("stroke=\"rgb(255,0,0)\" stroke-width=\"20\""),
      "red tick marks as thick as the axis: {svg}"
    );
  }
}

mod sampling {
  use super::*;

  /// WL samples the domain's ends a millionth of a step inside it.
  #[test]
  fn ends_are_sampled_just_inside() {
    assert_eq!(
      interpret(
        "ToString[Min[Reap[Plot[x, {x, 0, 5}, EvaluationMonitor :> Sow[x]]][[2, 1]]], InputForm]"
      )
      .unwrap(),
      "1.0204081632653061*^-7"
    );
  }

  /// So a step function ends flat on its last step instead of jumping to
  /// the value at the very end.
  #[test]
  fn step_function_without_exclusions_ends_flat() {
    assert_eq!(
      interpret("PlotRange[Plot[Floor[x], {x, 0, 5}, Exclusions -> None]]")
        .unwrap(),
      "{{0., 5.}, {0., 4.}}"
    );
  }
}

mod styles {
  use super::*;

  /// A grid line given its own color keeps the default grid opacity.
  #[test]
  fn styled_grid_line_keeps_default_opacity() {
    let svg = svg_of(
      "Plot[x^2, {x, 0, 3}, GridLines -> {{{2, Directive[Blue, Thick]}}, None}]",
    );
    assert!(svg.contains("opacity=\"0.4\" stroke=\"#0000FF\""), "{svg}");
  }

  /// `Dashing[Small]` is WL's 2.5 on, 4.5 off, not `{Small, Small}`.
  #[test]
  fn named_dashing_size() {
    let svg = svg_of("Plot[x, {x, 0, 1}, PlotStyle -> Dashing[Small]]");
    assert!(svg.contains("stroke-dasharray=\"25.0,45.0\""), "{svg}");
    let svg = svg_of(
      "Plot[Tan[x], {x, 0, 10}, Exclusions -> {Cos[x] == 0}, \
       ExclusionsStyle -> Dashing[Small]]",
    );
    assert!(svg.contains("stroke-dasharray=\"25.0,45.0\""), "{svg}");
  }

  /// Curves are drawn in list order whether or not they are dashed, so a
  /// solid curve listed after dashed ones covers them.
  #[test]
  fn dashed_curves_keep_their_drawing_order() {
    let strokes = |code: &str| -> Vec<String> {
      svg_of(code)
        .lines()
        .filter(|l| l.starts_with("<polyline"))
        .filter_map(|l| {
          let at = l.find(" stroke=\"")? + 9;
          Some(l[at..at + l[at..].find('"')?].to_string())
        })
        .collect()
    };
    assert_eq!(
      strokes(
        "Plot[{Sin[x], Cos[x], Sin[x] + Cos[x]}, {x, 0, 1}, \
         PlotStyle -> {Dashed, Dashed, Thick}]"
      ),
      ["rgb(61,153,204)", "rgb(242,160,36)", "#74B336"]
    );
    assert_eq!(
      strokes(
        "Plot[{x, 2 x, 3 x}, {x, 0, 1}, PlotStyle -> {Red, Dashed, Blue}]"
      ),
      ["#FF0000", "rgb(242,160,36)", "#0000FF"]
    );
    // A dashed grid line lies under the curve.
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, GridLines -> {{0.5}, None}, \
       GridLinesStyle -> Dashed, PlotStyle -> Red]",
    );
    let grid = svg.find("stroke-dasharray").unwrap();
    let curve = svg.find("stroke=\"#FF0000\"").unwrap();
    assert!(grid < curve, "{svg}");
  }

  /// `TraditionalForm` sets single-letter symbols in italics; a string
  /// stays upright.
  #[test]
  fn traditional_form_italicizes_single_letters() {
    let markup = |code: &str| {
      let expr = woxi::interpret_to_expr(code).unwrap();
      woxi::functions::graphics::expr_to_svg_markup(&expr)
    };
    assert_eq!(
      markup("TraditionalForm[Sin[x]]"),
      "sin(<tspan font-style=\"italic\">x</tspan>)"
    );
    assert!(!markup("TraditionalForm[\"x\"]").contains("italic"));
  }
}

/// An automatically sized `GraphicsRow` of several plots fits about 640
/// points (WL: 645 for three), each plot laid out at its share.
#[test]
fn graphics_row_fits_640_points() {
  let svg = svg_of(
    "GraphicsRow[{Plot[Sin[x], {x, 0, 2 Pi}], PolarPlot[Cos[3 t], {t, 0, Pi}], \
     ParametricPlot[{Cos[t], Sin[2 t]}, {t, 0, 2 Pi}]}]",
  );
  let width = root_attr(&svg, "width");
  assert!((600.0..=680.0).contains(&width), "{width}");
}

#[test]
fn default_plot_axes_use_gray_hairlines_with_any_background() {
  for size in [240, 480] {
    for background in ["White", "LightYellow"] {
      let svg = svg_of(&format!(
        "Plot[Sin[x], {{x, -2, 2}}, ImageSize -> {size}, Background -> {background}]"
      ));
      // Plot SVGs use ten render units per display point: WL's 0.2 pt
      // axis stroke is two units, independently of image size/background.
      let axes: Vec<_> =
        svg.lines().filter(|l| l.starts_with("<line ")).collect();
      assert!(!axes.is_empty(), "{svg}");
      assert!(axes.iter().all(|l| l.contains("stroke=\"rgb(102,102,102)\" stroke-width=\"2\"")), "{svg}");
    }
  }
}

#[test]
fn graphics_axes_and_frames_use_gray_unless_explicitly_styled() {
  for decoration in ["Axes", "Frame"] {
    for primitive in ["Disk[]", "Rectangle[{-2, -1}, {3, 2}]"] {
      let svg = svg_of(&format!("Graphics[{primitive}, {decoration} -> True]"));
      let width = if decoration == "Frame" { "0.5" } else { "0.2" };
      assert!(
        svg.contains(&format!("stroke=\"#666666\" stroke-width=\"{width}\"")),
        "{svg}"
      );
      let styled = svg_of(&format!(
        "Graphics[{primitive}, {decoration} -> True, {decoration}Style -> Red]"
      ));
      assert!(styled.contains("stroke=\"rgb(255,0,0)\""), "{styled}");
      assert!(!styled.contains("stroke=\"#666666\""), "{styled}");
    }
  }
}
