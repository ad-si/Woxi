//! Plot layout and filling details checked against wolframscript (WL 15):
//! how a fixed `ImageSize` holds the plotting area, and how `FillingStyle`
//! and `ColorFunction` paint a fill.
use super::*;

fn svg_of(expr: &str) -> String {
  interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
}

/// The `width`/`height` attributes of the root `<svg>` element.
fn svg_size(svg: &str) -> (f64, f64) {
  let tag = &svg[..svg.find('>').unwrap()];
  let attr = |name: &str| -> f64 {
    let start = tag.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
    let end = start + tag[start..].find('"').unwrap();
    tag[start..end].parse().unwrap()
  };
  (attr("width"), attr("height"))
}

/// The points of every `<polyline>` drawn with the given stroke color.
fn polyline_points(svg: &str, stroke: &str) -> Vec<(f64, f64)> {
  svg
    .split("<polyline ")
    .skip(1)
    .filter(|tag| tag.contains(&format!("stroke=\"{stroke}\"")))
    .flat_map(|tag| {
      let start = tag.find("points=\"").unwrap() + 8;
      let end = start + tag[start..].find('"').unwrap();
      tag[start..end]
        .split_whitespace()
        .map(|p| {
          let (x, y) = p.split_once(',').unwrap();
          (x.parse().unwrap(), y.parse().unwrap())
        })
        .collect::<Vec<_>>()
    })
    .collect()
}

/// Width and height of the bounding box of the default-colored curve.
fn curve_extent(svg: &str) -> (f64, f64) {
  let pts = polyline_points(svg, "#3D99CC");
  assert!(!pts.is_empty(), "no curve in {svg}");
  let (mut x0, mut x1, mut y0, mut y1) = (
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
  );
  for (x, y) in pts {
    (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
  }
  (x1 - x0, y1 - y0)
}

mod image_size {
  use super::*;

  // `ImageSize -> {w, h}` fixes the image, but the plotting area keeps its
  // default 1/GoldenRatio shape inside it (WL centers it vertically); it
  // does not stretch to the square.
  #[test]
  fn fixed_size_keeps_default_aspect_ratio() {
    let svg = svg_of("Plot[x, {x, 0, 1}, ImageSize -> {200, 200}]");
    assert_eq!(svg_size(&svg), (200.0, 200.0));
    let (w, h) = curve_extent(&svg);
    // The curve spans the whole x range and the y range minus the default
    // padding of 4 % on each side: 1/GoldenRatio / 1.08.
    let ratio = h / w;
    assert!((ratio - 0.618_034 / 1.08).abs() < 0.01, "ratio {ratio}");
  }

  // `AspectRatio -> Full` stretches the plotting area over the fixed image.
  #[test]
  fn aspect_ratio_full_fills_fixed_size() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, ImageSize -> {200, 200}, AspectRatio -> Full]",
    );
    assert_eq!(svg_size(&svg), (200.0, 200.0));
    let (w, h) = curve_extent(&svg);
    assert!(h / w > 0.8, "ratio {}", h / w);
  }

  // An explicit `AspectRatio` also fits inside a fixed image instead of
  // resizing it.
  #[test]
  fn explicit_aspect_ratio_fits_fixed_size() {
    let svg =
      svg_of("Plot[x, {x, 0, 1}, ImageSize -> {300, 200}, AspectRatio -> 1/4]");
    assert_eq!(svg_size(&svg), (300.0, 200.0));
    let (w, h) = curve_extent(&svg);
    assert!((h / w - 0.25 / 1.08).abs() < 0.01, "ratio {}", h / w);
  }

  // `ImageSize -> {Automatic, h}` keeps the height; with a tall
  // `AspectRatio` the width shrinks to match.
  #[test]
  fn fixed_height_derives_width_from_aspect_ratio() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, ImageSize -> {Automatic, 150}, AspectRatio -> 2]",
    );
    let (width, height) = svg_size(&svg);
    assert_eq!(height, 150.0);
    // WL: about 105 px (a 60 px wide plotting area plus the tick labels);
    // it used to come out 243 px wide, from the default aspect ratio.
    assert!(width < 130.0, "width {width}");
    let (w, h) = curve_extent(&svg);
    assert!((h / w - 2.0 / 1.08).abs() < 0.05, "ratio {}", h / w);
  }

  // A bare width still derives the height from the aspect ratio.
  #[test]
  fn width_only_still_follows_aspect_ratio() {
    let svg = svg_of("Plot[x, {x, 0, 1}, ImageSize -> 300, AspectRatio -> 1]");
    let (width, height) = svg_size(&svg);
    assert_eq!(width, 300.0);
    assert!(height > 250.0, "height {height}");
  }
}

mod filling_style {
  use super::*;

  // `FillingStyle -> {below, above}`: the first color fills where the curve
  // lies below the filling level, the second where it lies above.
  #[test]
  fn two_colors_are_below_then_above() {
    let below = svg_of(
      "Plot[-1 - x^2, {x, 0, 1}, Filling -> Axis, FillingStyle -> {Red, Blue}]",
    );
    assert!(below.contains("fill=\"#FF0000\""), "{below}");
    assert!(!below.contains("fill=\"#0000FF\""), "{below}");
    let above = svg_of(
      "Plot[1 + x^2, {x, 0, 1}, Filling -> Axis, FillingStyle -> {Red, Blue}]",
    );
    assert!(above.contains("fill=\"#0000FF\""), "{above}");
    assert!(!above.contains("fill=\"#FF0000\""), "{above}");
  }

  // A sign-changing curve gets both colors.
  #[test]
  fn two_colors_split_at_the_axis() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 2 Pi}, Filling -> Axis, FillingStyle -> {Red, Blue}]",
    );
    assert!(svg.contains("fill=\"#FF0000\""), "{svg}");
    assert!(svg.contains("fill=\"#0000FF\""), "{svg}");
  }

  // Between two curves, `{{j}, {below, above}}` uses the same order: with
  // `Sin` below `Cos` near 0 the first color is drawn there.
  #[test]
  fn two_colors_between_curves_below_first() {
    let svg = svg_of(
      "Plot[{x, 2 x + 1}, {x, 0, 1}, Filling -> {1 -> {{2}, {Red, Blue}}}]",
    );
    assert!(svg.contains("fill=\"#FF0000\""), "{svg}");
    assert!(!svg.contains("fill=\"#0000FF\""), "{svg}");
  }
}

mod color_function_filling {
  use super::*;

  /// Opacities of the filled areas of a plot (`fill-opacity`/`opacity` of
  /// every filled `<polygon>`/`<path>` that is not the background).
  fn fill_opacities(svg: &str) -> Vec<f64> {
    svg
      .split('<')
      .filter(|t| t.starts_with("polygon ") || t.starts_with("path "))
      .filter(|t| t.contains("fill=\"#") || t.contains("fill=\"url(#"))
      .filter_map(|t| {
        let key = "opacity=\"";
        let start = t.find(key)? + key.len();
        let end = start + t[start..].find('"')?;
        t[start..end].parse().ok()
      })
      .collect()
  }

  // With a `ColorFunction` and the default `FillingStyle`, WL paints the
  // fill with the color function's colors at full strength.
  #[test]
  fn default_filling_is_opaque() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 2 Pi}, ColorFunction -> Function[{x, y}, Hue[y]], \
       Filling -> Axis]",
    );
    let ops = fill_opacities(&svg);
    assert!(!ops.is_empty(), "{svg}");
    assert!(ops.iter().all(|&o| o == 1.0), "{ops:?}");
  }

  // An explicit `FillingStyle -> Opacity[a]` drops the color function from
  // the fill: it is the series color at that opacity.
  #[test]
  fn explicit_opacity_uses_series_color() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 2 Pi}, ColorFunction -> Function[{x, y}, Hue[y]], \
       Filling -> Axis, FillingStyle -> Opacity[0.5]]",
    );
    assert!(
      svg.contains("fill=\"#3D99CC\" fill-opacity=\"0.5\"")
        || svg.contains("fill=\"#3D99CC\" opacity=\"0.5\"")
        || svg.contains("opacity=\"0.5\" fill=\"#3D99CC\""),
      "{svg}"
    );
  }

  // `ColorFunctionScaling` rescales over the sampled values, not the padded
  // plot range: the top of `Sin` gets (nearly) `Hue[1]`, pure red.
  #[test]
  fn scaling_uses_data_range() {
    let svg = svg_of(
      "Plot[Sin[x], {x, 0, 2 Pi}, ColorFunction -> Function[{x, y}, Hue[y]]]",
    );
    let reddest_blue = svg
      .split("stop-color=\"#FF00")
      .skip(1)
      .filter_map(|rest| u8::from_str_radix(rest.get(..2)?, 16).ok())
      .min()
      .expect("no magenta-red stroke");
    assert!(reddest_blue <= 4, "closest to red: #FF00{reddest_blue:02X}");
  }
}

mod frame_ticks {
  use super::*;

  // The frame labels every major tick inside the range, the last one
  // included: plotters' key-point search used to drop the top `1.0`.
  #[test]
  fn top_label_is_kept() {
    let svg = svg_of("Plot[Sin[x], {x, 0, 2 Pi}, Frame -> True]");
    for label in [
      ">\n-1.0\n<",
      ">\n-0.5\n<",
      ">\n0.0\n<",
      ">\n0.5\n<",
      ">\n1.0\n<",
    ] {
      assert!(svg.contains(label), "missing {label} in {svg}");
    }
  }
}

mod grid_lines {
  use super::*;

  // WL 15: `GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]]`.
  #[test]
  fn default_style_is_wl15_gray() {
    let svg = svg_of("Plot[x, {x, 0, 1}, GridLines -> Automatic]");
    assert!(svg.contains("stroke=\"#808080\""), "{svg}");
  }
}

mod list_polar_plot {
  use super::*;

  /// Centers of the `<circle>` data points.
  fn circles(svg: &str) -> Vec<(f64, f64)> {
    svg
      .split("<circle ")
      .skip(1)
      .map(|tag| {
        let attr = |name: &str| -> f64 {
          let key = format!("{name}=\"");
          let start = tag.find(&key).unwrap() + key.len();
          let end = start + tag[start..].find('"').unwrap();
          tag[start..end].parse().unwrap()
        };
        (attr("cx"), attr("cy"))
      })
      .collect()
  }

  // `{θ, r}` pairs are placed at their own angle, and the points are not
  // joined by default (WL draws `Point`s): angles 0, Pi/2 and Pi at radius
  // 1 are (1, 0), (0, 1) and (-1, 0).
  #[test]
  fn pairs_use_their_angle_and_are_points() {
    let svg = svg_of("ListPolarPlot[{{0, 1}, {Pi/2, 1}, {Pi, 1}}]");
    let pts = circles(&svg);
    assert_eq!(pts.len(), 3, "{svg}");
    assert!((pts[0].1 - pts[2].1).abs() < 1.0, "{pts:?}");
    assert!(pts[1].1 < pts[0].1 - 100.0, "{pts:?}");
    assert!(pts[0].0 > pts[1].0 && pts[1].0 > pts[2].0, "{pts:?}");
  }

  // `Joined -> True` draws a line instead of points.
  #[test]
  fn joined_draws_a_line() {
    let svg = svg_of("ListPolarPlot[{1, 2, 3, 4}, Joined -> True]");
    assert!(circles(&svg).is_empty(), "{svg}");
    assert_eq!(polyline_points(&svg, "#3D99CC").len(), 4, "{svg}");
  }
}

mod sampling {
  use super::*;

  fn evaluations(plot: &str) -> i64 {
    interpret(&format!("Length[Reap[{plot}][[2, 1]]]"))
      .unwrap()
      .parse()
      .unwrap()
  }

  // WL starts from 50 points; `MaxRecursion -> 0` keeps exactly those.
  #[test]
  fn max_recursion_zero_keeps_the_initial_points() {
    assert_eq!(
      evaluations(
        "Plot[Sin[1/x], {x, 0.001, 0.1}, MaxRecursion -> 0, \
         EvaluationMonitor :> Sow[x]]"
      ),
      50
    );
    let more = evaluations(
      "Plot[Sin[1/x], {x, 0.001, 0.1}, MaxRecursion -> 3, \
       EvaluationMonitor :> Sow[x]]",
    );
    let most = evaluations(
      "Plot[Sin[1/x], {x, 0.001, 0.1}, EvaluationMonitor :> Sow[x]]",
    );
    assert!(50 < more && more < most, "{more} {most}");
  }

  // A rapidly oscillating curve is refined until its peaks are reached
  // (it used to stop at a 500-point budget, cutting most peaks short).
  #[test]
  fn rapid_oscillation_reaches_its_peaks() {
    let svg = svg_of("Plot[Sin[30 Sin[x]], {x, 0, 10}]");
    let pts = polyline_points(&svg, "#3D99CC");
    // The extrema of the curve in SVG y (down) and the count of local
    // maxima that come within 2 % of the top.
    let top = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let bottom = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let near_top = |y: f64| y < top + (bottom - top) * 0.02;
    let peaks = pts
      .windows(3)
      .filter(|w| w[1].1 <= w[0].1 && w[1].1 <= w[2].1 && w[1].1 < w[0].1)
      .collect::<Vec<_>>();
    // (At x = Pi/2 and 5 Pi/2 the curve has genuine low maxima, Sin[30] =
    // -0.988, in its lower half; every maximum in the upper half is a 1.)
    let upper_half = |y: f64| y < f64::midpoint(top, bottom);
    let short = peaks
      .iter()
      .filter(|w| upper_half(w[1].1) && !near_top(w[1].1))
      .count();
    assert!(peaks.len() > 30, "{} peaks", peaks.len());
    assert_eq!(short, 0, "{short} of {} peaks fall short", peaks.len());
  }

  // Where the function has no real value at all there is no boundary to
  // look for, so the sampler does not keep halving those intervals.
  #[test]
  fn no_refinement_where_the_curve_does_not_exist() {
    let n =
      evaluations("Plot[Sqrt[x], {x, -10, 1}, EvaluationMonitor :> Sow[x]]");
    assert!(n < 150, "{n} evaluations");
  }
}

// WL 15 draws the axes of a framed plot as hairlines through the origin,
// where the origin lies strictly inside the unpadded plot range.
mod framed_axes {
  use super::*;

  fn axis_lines(expr: &str) -> usize {
    svg_of(expr)
      .matches("stroke=\"#666666\" stroke-width=\"2\" points=")
      .count()
  }

  #[test]
  fn both_axes_inside_the_frame() {
    assert_eq!(axis_lines("Plot[x, {x, -1, 1}, Frame -> True]"), 2);
  }

  #[test]
  fn an_origin_at_or_beyond_the_data_range_draws_no_axis() {
    // y = 0 inside, x = 0 at the left end of the range: only the x axis.
    assert_eq!(axis_lines("Plot[Cos[x], {x, 0, 2 Pi}, Frame -> True]"), 1);
    assert_eq!(axis_lines("Plot[x + 2, {x, 1, 3}, Frame -> True]"), 0);
    assert_eq!(
      axis_lines("ListPlot[Table[Sin[x], {x, 0, 2 Pi, Pi/30}], Frame -> True]"),
      1
    );
  }

  #[test]
  fn axes_false_draws_none() {
    assert_eq!(
      axis_lines("Plot[x, {x, -1, 1}, Frame -> True, Axes -> False]"),
      0
    );
  }
}

// ListPolarPlot draws its points at ListPlot's automatic size (WL 15:
// PointSize[6 11/6000] for 63 points, PointSize[7 11/6000] for 5).
#[test]
fn list_polar_plot_uses_the_list_plot_point_size() {
  let radius = |expr: &str| -> String {
    let svg = svg_of(expr);
    let tag = &svg[svg.find("<circle ").unwrap()..];
    let start = tag.find(" r=\"").unwrap() + 4;
    tag[start..start + tag[start..].find('"').unwrap()].to_string()
  };
  let pts = "Table[{x, Sqrt[x]}, {x, 0, 2 Pi, 0.1}]";
  assert_eq!(
    radius(&format!("ListPolarPlot[{pts}]")),
    radius(&format!("ListPlot[{pts}]"))
  );
  assert_ne!(
    radius("ListPolarPlot[{1, 2, 3, 2, 1}]"),
    radius(&format!("ListPolarPlot[{pts}]"))
  );
}

// `Callout[data, label]` on a whole curve or dataset hangs its label off a
// horizontal gray leader at the series' end; `Labeled[data, label]` puts
// it right after the last point; expression labels are TraditionalForm.
mod series_end_labels {
  use super::*;

  /// `(x, y)` of the `<text>` holding exactly `label`.
  fn label_at(svg: &str, label: &str) -> (f64, f64) {
    let svg = &super::super::graphics::without_italics(svg);
    let end = svg.find(&format!(">{label}</text>")).unwrap();
    let tag = &svg[svg[..end].rfind("<text ").unwrap()..end];
    let attr = |name: &str| -> f64 {
      let start = tag.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
      tag[start..start + tag[start..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    (attr("x"), attr("y"))
  }

  #[test]
  fn callout_and_labeled_datasets() {
    let svg = svg_of(
      "ListPlot[{Callout[Range[5], PartitionsQ[n]], Labeled[Range[5] + 3, \
       \"lab\"]}]",
    );
    // One leader, for the callout; the label is typeset q(n).
    assert_eq!(svg.matches("fill=\"none\" stroke=\"#AAAAAA\"").count(), 1);
    let (qx, _) = label_at(&svg, "q(n)");
    let (lx, _) = label_at(&svg, "lab");
    // Both sit right of the last points, the callout's past its leader.
    assert!(qx > lx + 200.0, "{qx} {lx}");
  }

  #[test]
  fn plot_callout_at_the_curve_end() {
    let svg = svg_of("Plot[{Callout[x, x], Callout[2 x, \"two\"]}, {x, 0, 1}]");
    let (x1, y1) = label_at(&svg, "x");
    let (x2, y2) = label_at(&svg, "two");
    assert!((x1 - x2).abs() < 1e-6);
    // 2 x ends higher (smaller SVG y).
    assert!(y2 < y1, "{y1} {y2}");
  }

  #[test]
  fn close_labels_are_spread_apart() {
    let svg = svg_of(
      "ListPlot[{Callout[{1, 2, 3}, \"a\"], Callout[{1, 2, 3.01}, \"b\"]}]",
    );
    let (_, ya) = label_at(&svg, "a");
    let (_, yb) = label_at(&svg, "b");
    assert!((ya - yb).abs() > 150.0, "{ya} {yb}");
  }

  #[test]
  fn plot_labels_hang_off_leaders_in_traditional_form() {
    let svg = svg_of(
      "Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotLabels -> {\"sine\", None}]",
    );
    assert!(svg.contains(">sine</text>"));
    assert_eq!(svg.matches("fill=\"none\" stroke=\"#AAAAAA\"").count(), 1);
    let svg = svg_of("Plot[x^2, {x, 0, 1}, PlotLabels -> {x^2}]");
    assert!(svg.contains("baseline-shift=\"super\""), "{svg}");
  }

  #[test]
  fn a_positioned_callout_keeps_its_place() {
    // `Callout[data, label, Above]` is not moved to the series end: it
    // hangs above the highest point on one gray leader (no end leader).
    let svg = svg_of("ListPlot[Callout[{1, 3, 2}, \"top\", Above]]");
    assert!(svg.contains(">top</text>"));
    assert_eq!(svg.matches("<polyline points=").count(), 0);
    assert_eq!(svg.matches("fill=\"none\" stroke=\"#AAAAAA\"").count(), 1);
  }
}

// `ListPlot[{a, Legended[b, label], c}]` gets a point legend with the one
// entry, in the color of the series it names (WL: `Legended[g,
// Placed[PointLegend[…], After]]`).
mod legended_datasets {
  use super::*;

  #[test]
  fn legended_dataset_gets_a_point_legend_entry() {
    let svg = svg_of(
      "ListPlot[{{1, 2, 1}, Legended[{2, 3, 2}, \"average\"], {3, 4, 3}}]",
    );
    assert!(svg.contains(">average</text>"), "{svg}");
    // One swatch, a point in the second (orange) palette color.
    assert_eq!(
      svg.matches("r=\"25.0\" fill=\"rgb(242,160,36)\"/>").count(),
      1
    );
    assert_eq!(svg.matches("r=\"25.0\" fill=").count(), 1);
  }

  #[test]
  fn placed_below_puts_the_legend_under_the_plot() {
    let right =
      svg_size(&svg_of("ListPlot[{{1, 2}, Legended[{2, 3}, \"avg\"]}]"));
    let below = svg_size(&svg_of(
      "ListPlot[{{1, 2}, Legended[{2, 3}, Placed[\"avg\", Below]]}]",
    ));
    assert!(right.0 > 360.0, "{right:?}");
    assert_eq!(below.0, 360.0);
    assert!(below.1 > right.1, "{below:?} {right:?}");
  }

  #[test]
  fn list_line_plot_legended_dataset_draws_a_line_swatch() {
    let svg = svg_of("ListLinePlot[{{1, 2, 1}, Legended[{2, 3, 2}, \"b\"]}]");
    assert!(svg.contains(">b</text>"), "{svg}");
    assert!(svg.contains("stroke=\"rgb(242,160,36)\""), "{svg}");
  }
}

// `ImageSize -> Full | Scaled[s]` sizes a plot from the `Pane` holding it,
// and `Framed` draws WL's square black frame 5 points around it.
mod pane_relative_size {
  use super::*;

  /// The `width`/`height` of the first nested `<svg>` (the framed content).
  fn nested_size(svg: &str) -> (f64, f64) {
    let rest = &svg[svg.find('>').unwrap()..];
    svg_size(&rest[rest.find("<svg").unwrap()..])
  }

  #[test]
  fn scaled_takes_that_fraction_of_the_pane_width() {
    let svg = svg_of(
      "Pane[Plot[Sin[x], {x, 0, 10}, ImageSize -> Scaled[0.5]], {200, 100}]",
    );
    // The pane is the whole box; the plot is half its width, its height
    // following the aspect ratio (WL: 100 x 52.9), in the top-left corner.
    assert_eq!(svg_size(&svg), (200.0, 100.0));
    assert!(svg.contains("translate(0.0 0.0)"), "{svg}");
    let (w, h) = nested_size(&svg);
    assert_eq!(w, 100.0);
    assert!((50.0..56.0).contains(&h), "{h}");
  }

  #[test]
  fn full_fills_the_pane() {
    let svg =
      svg_of("Pane[Plot[Sin[x], {x, 0, 10}, ImageSize -> Full], {200, 100}]");
    assert_eq!(svg_size(&svg), (200.0, 100.0));
    let (w, h) = curve_extent(&svg);
    // The curve spans (nearly) the whole pane, its aspect ratio dropped.
    assert!(w > 1300.0 && h > 650.0, "{w} x {h}");
  }

  #[test]
  fn framed_is_a_square_black_frame_5_points_out() {
    let svg = svg_of("Framed[Graphics[Disk[], ImageSize -> 50]]");
    assert_eq!(svg_size(&svg), (60.0, 60.0));
    assert!(svg.contains("rx=\"0.0\""), "{svg}");
    assert!(svg.contains("stroke=\"black\""), "{svg}");
    // A Panel keeps its rounded light-gray box.
    let panel = svg_of("Panel[Graphics[Disk[], ImageSize -> 50]]");
    assert!(!panel.contains("stroke=\"black\""), "{panel}");
  }
}

// `PlotLayout`: one framed panel per curve, all in the first palette color
// over one shared plot range, ticks only along the outer edges (WL 15).
mod panel_layout {
  use super::*;

  /// The `<rect … fill="none">` panel frames as `(x, y, w, h)`.
  fn frames(svg: &str) -> Vec<(f64, f64, f64, f64)> {
    svg
      .split("<rect ")
      .skip(1)
      .map(|t| &t[..t.find('>').unwrap()])
      .filter(|t| t.contains("fill=\"none\"") && t.contains("stroke=\"#"))
      .map(|t| {
        let t = format!(" {t}");
        let attr = |name: &str| -> f64 {
          let start = t.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
          let end = start + t[start..].find('"').unwrap();
          t[start..end].parse().unwrap()
        };
        (attr("x"), attr("y"), attr("width"), attr("height"))
      })
      .collect()
  }

  #[test]
  fn row_is_600_wide_with_one_panel_per_curve() {
    let svg = svg_of(
      "Plot[{Sin[x], Sin[2 x], Sin[3 x]}, {x, 0, 10}, PlotLayout -> \"Row\"]",
    );
    let (w, h) = svg_size(&svg);
    assert_eq!(w, 600.0);
    assert!((130.0..155.0).contains(&h), "{h}");
    let f = frames(&svg);
    assert_eq!(f.len(), 3);
    // Same size, one row, 5 pt apart, golden-ratio plot areas.
    for p in &f {
      assert!((p.1 - f[0].1).abs() < 1e-6);
      assert!((p.2 - f[0].2).abs() < 1e-6);
      assert!((p.3 / p.2 - 0.618).abs() < 0.01);
    }
    assert!((f[1].0 - (f[0].0 + f[0].2) - 5.0).abs() < 0.01);
    // All curves in the first palette color.
    assert!(svg.contains("#3D99CC"));
    assert!(!svg.contains("#F2A024"));
    // x ticks 0, 2.5, …, 10 under each panel; the second and third panels
    // drop their "0.0", which would run into the previous panel's "10.0"
    // (the other "0.0" is the y label).
    assert_eq!(svg.matches(">2.5</text>").count(), 3);
    assert_eq!(svg.matches(">0.0</text>").count(), 2);
  }

  #[test]
  fn column_stacks_the_panels_with_x_labels_only_at_the_bottom() {
    let svg =
      svg_of("Plot[{Sin[x], Cos[x]}, {x, 0, 4 Pi}, PlotLayout -> \"Column\"]");
    let (w, _) = svg_size(&svg);
    assert_eq!(w, 360.0);
    let f = frames(&svg);
    assert_eq!(f.len(), 2);
    assert!((f[0].0 - f[1].0).abs() < 1e-6);
    assert!((f[1].1 - (f[0].1 + f[0].3) - 5.0).abs() < 0.01);
    // y labels on both panels, x labels (0, 5, 10) once.
    assert_eq!(svg.matches(">0.5</text>").count(), 2);
    assert_eq!(svg.matches(">5</text>").count(), 1);
  }

  #[test]
  fn counted_layouts() {
    let fs = "Table[BesselJ[n, x], {n, 0, 5}], {x, -20, 20}";
    // {"Row", 2}: two rows of three. {"Column", 4}: four columns holding
    // 2, 2, 1, 1 panels. {"Column", UpTo[4]}: three columns of two.
    for (layout, rows, cols) in [
      ("{\"Row\", 2}", 2, 3),
      ("{\"Column\", 4}", 2, 4),
      ("{\"Column\", UpTo[4]}", 2, 3),
    ] {
      let svg =
        svg_of(&format!("Plot[Evaluate[{fs}], PlotLayout -> {layout}]"));
      let f = frames(&svg);
      assert_eq!(f.len(), 6, "{layout}");
      let mut xs: Vec<i64> = f.iter().map(|p| p.0.round() as i64).collect();
      let mut ys: Vec<i64> = f.iter().map(|p| p.1.round() as i64).collect();
      xs.sort_unstable();
      xs.dedup();
      ys.sort_unstable();
      ys.dedup();
      assert_eq!((ys.len(), xs.len()), (rows, cols), "{layout}");
    }
  }

  #[test]
  fn panels_take_the_plot_style_as_one_style() {
    // WL collapses `PlotStyle -> {Red, Green}` per panel: both are green.
    let svg = svg_of(
      "Plot[{Sin[x], Cos[x]}, {x, 0, 1}, PlotLayout -> \"Row\", \
       PlotStyle -> {Red, Green}]",
    );
    assert!(!svg.contains("#FF0000"));
    assert_ne!(polyline_points(&svg, "#00FF00"), [] as [(f64, f64); 0]);
  }

  #[test]
  fn list_plot_panels_share_the_plot_range() {
    let svg = svg_of(
      "ListPlot[{{10, 9, 4, 3, 5, 3, 5, 5, 2, 6}, {2, 10, 5, 6, 9, 4, 9, 3, \
       7, 2}, {7, 8, 3, 4, 4, 2, 6, 3, 8, 7}}, PlotLayout -> \"Column\", \
       Joined -> True, ImageSize -> Medium]",
    );
    // A named size caps the height at 1.2 times its width (432), so the
    // column comes out narrower than 360.
    let (w, h) = svg_size(&svg);
    assert!(h <= 433.0 && w < 300.0, "{w} x {h}");
    // Every panel labels 0.0 … 10.0 on its y axis.
    assert_eq!(svg.matches(">10.0</text>").count(), 4);
    assert_eq!(svg.matches(">7.5</text>").count(), 4);
  }
}

/// The automatic `ImagePadding`: the room the frame, the axes and their
/// labels take around the plotting area, and so the picture's height.
/// Every size here is what wolframscript (WL 15) exports.
mod automatic_image_padding {
  use super::*;

  fn size(expr: &str) -> (f64, f64) {
    svg_size(&svg_of(expr))
  }

  #[test]
  fn framed_plots_pad_for_their_tick_labels() {
    // Left: the widest y tick label plus 3 pt; bottom: a label line plus
    // 3 pt plus the extra point every bottom edge keeps.
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Frame -> True]"),
      (360.0, 223.0)
    );
    assert_eq!(size("Plot[x, {x, 0, 1}, Frame -> True]"), (360.0, 227.0));
    assert_eq!(size("Plot[x, {x, 0, 1000}, Frame -> True]"), (360.0, 220.0));
    // `10 000` is set in digit blocks, a little wider still.
    assert_eq!(
      size("Plot[x, {x, 0, 10000}, Frame -> True]"),
      (360.0, 213.0)
    );
    assert_eq!(size("Plot[x, {x, -10, 10}, Frame -> True]"), (360.0, 225.0));
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Frame -> True, ImageSize -> 600]"),
      (600.0, 372.0)
    );
  }

  #[test]
  fn frame_ticks_pad_each_edge_they_label() {
    let sin = "Table[Sin[x], {x, 0, 2 Pi, Pi/180}]";
    // `All` labels the top and right edges too: the frame moves in to
    // x = 23.2..335.8, y = 14.17..207.36.
    let svg = svg_of(&format!(
      "ListPlot[{sin}, Frame -> True, FrameTicks -> All]"
    ));
    assert_eq!(svg_size(&svg), (360.0, 223.0));
    let frame = polyline_points(&svg, "#666666");
    assert!(
      frame.iter().any(|&(x, y)| x == 3357.0 && y == 142.0),
      "{frame:?}"
    );
    // Only the bottom edge labeled.
    assert_eq!(
      size(&format!(
        "ListPlot[{sin}, Frame -> True, FrameTicks -> {{{{None, None}}, \
         {{Automatic, None}}}}]"
      )),
      (360.0, 237.0)
    );
    // Marks reaching 0.1 of the width out of the top edge push its labels
    // up by as much.
    assert_eq!(
      size(&format!(
        "ListPlot[{sin}, Frame -> True, FrameTicks -> {{{{Automatic, \
         Automatic}}, {{{{{{50, a, 0.1}}, {{180, b, 0.1}}}}, {{{{50, a, \
         {{0.2, 0.1}}}}, {{180, b, {{0.2, 0.1}}}}}}}}}}]"
      )),
      (360.0, 271.0)
    );
  }

  #[test]
  fn end_labels_overhanging_a_small_frame_widen_its_padding() {
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Frame -> True, ImageSize -> Small]"),
      (180.0, 111.0)
    );
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, ImageSize -> Tiny]"),
      (100.0, 53.0)
    );
  }

  #[test]
  fn labels_add_their_own_room() {
    assert_eq!(
      size(
        "Plot[Sin[x], {x, 0, 10}, Frame -> True, FrameLabel -> {\"x\", \"y\"}]"
      ),
      (360.0, 230.0)
    );
    assert_eq!(
      size(
        "Plot[Sin[x], {x, 0, 10}, Frame -> True, \
         FrameLabel -> {{\"l\", \"r\"}, {\"b\", \"t\"}}]"
      ),
      (360.0, 236.0)
    );
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, PlotLabel -> \"Title\"]"),
      (360.0, 227.0)
    );
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Frame -> True, PlotLabel -> \"Title\"]"),
      (360.0, 237.0)
    );
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, AxesLabel -> {\"x\", \"y\"}]"),
      (360.0, 223.0)
    );
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, AxesLabel -> {\"time\", \"amplitude\"}]"),
      (360.0, 214.0)
    );
  }

  #[test]
  fn unframed_plots_pad_only_where_the_axes_labels_stick_out() {
    assert_eq!(size("Plot[Sin[x], {x, 0, 10}]"), (360.0, 214.0));
    assert_eq!(size("Plot[x^2, {x, -3, 3}]"), (360.0, 226.0));
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Ticks -> None]"),
      (360.0, 224.0)
    );
    // No axes: no padding at all.
    assert_eq!(
      size("Plot[Sin[x], {x, 0, 10}, Axes -> False]"),
      (360.0, 223.0)
    );
    assert_eq!(
      size("ParametricPlot[{2 Cos[t], Sin[t]}, {t, 0, 2 Pi}]"),
      (360.0, 181.0)
    );
  }

  #[test]
  fn list_plots() {
    assert_eq!(size("ListPlot[Range[10]]"), (360.0, 229.0));
    assert_eq!(size("ListPlot[{1, 4, 9}]"), (360.0, 232.0));
    assert_eq!(size("ListPlot[{1, 4, 9}, Frame -> True]"), (360.0, 232.0));
    assert_eq!(
      size("ListLinePlot[{3, 1, 4, 1, 5, 9, 2, 6}]"),
      (360.0, 232.0)
    );
    assert_eq!(
      size("StackedListPlot[{{1, 2, 3}, {2, 3, 1}}]"),
      (360.0, 232.0)
    );
  }

  #[test]
  fn log_plots_pad_their_range_like_plot() {
    assert_eq!(size("LogPlot[Exp[x], {x, 0, 10}]"), (360.0, 216.0));
  }

  #[test]
  fn graphics_with_axes_or_a_frame_keep_their_image_width() {
    assert_eq!(
      size(
        "Graphics[Line[{{0, -1}, {10, 1}}], Axes -> True, \
         AspectRatio -> 1/GoldenRatio, ImageSize -> 380]"
      ),
      (380.0, 227.0)
    );
    assert_eq!(
      size(
        "Graphics[Line[{{0, -1}, {10, 1}}], Frame -> True, \
         AspectRatio -> 1/GoldenRatio, ImageSize -> 380]"
      ),
      (380.0, 236.0)
    );
    // An automatic size is 360 wide, axes and labels included.
    assert_eq!(
      size("Graphics[Line[{{0, -1}, {10, 1}}], Axes -> True]"),
      (360.0, 81.0)
    );
    assert_eq!(
      size("Graphics[Line[{{0, -1}, {10, 1}}], Frame -> True]"),
      (360.0, 93.0)
    );
    assert_eq!(size("Graphics[Circle[], Frame -> True]"), (360.0, 351.0));
    assert_eq!(
      size("Graphics[Disk[], Frame -> True, FrameLabel -> {\"x\", \"y\"}]"),
      (360.0, 351.0)
    );
  }

  #[test]
  fn bare_graphics_keep_a_point_right_and_below() {
    // The y range is padded by 0.02 of the x span, the x range by 1/48.
    assert_eq!(size("Graphics[Circle[]]"), (360.0, 360.0));
    assert_eq!(size("Graphics[Rectangle[{0, 0}, {2, 1}]]"), (360.0, 188.0));
    assert_eq!(size("Graphics[Rectangle[{0, 0}, {3, 1}]]"), (360.0, 130.0));
    assert_eq!(
      size("Graphics[Rectangle[{0, 0}, {3, 1}], ImageSize -> 200]"),
      (200.0, 73.0)
    );
    // Too tall for 1.2 × 360: the height is capped, the width follows.
    assert_eq!(size("Graphics[Rectangle[{0, 0}, {1, 3}]]"), (149.0, 432.0));
    assert_eq!(
      size(
        "Graphics[Line[{{0, -1}, {10, 1}}], AspectRatio -> 1/GoldenRatio, \
         ImageSize -> 380]"
      ),
      (380.0, 236.0)
    );
    // A title band replaces that point.
    assert_eq!(
      size("Graphics[Disk[], PlotLabel -> \"Title\"]"),
      (360.0, 373.0)
    );
    assert_eq!(
      size("Graphics[Rectangle[{0, 0}, {2, 1}], PlotLabel -> \"T\"]"),
      (360.0, 201.0)
    );
  }

  #[test]
  fn frame_tick_labels_sit_three_points_outside_the_frame() {
    // `Plot[Sin[x], {x, 0, 10}, Frame -> True]`: the y labels end 3 pt left
    // of the frame at x = 23.2.
    let svg = svg_of("Plot[Sin[x], {x, 0, 10}, Frame -> True]");
    let label = svg
      .split("<text ")
      .find(|t| {
        t.contains("text-anchor=\"end\"")
          && t.split_once('>').is_some_and(|(_, text)| {
            text.split('<').next().map(str::trim) == Some("1.0")
          })
      })
      .expect("no 1.0 label");
    let x: f64 = label
      .split("x=\"")
      .nth(1)
      .and_then(|r| r.split('"').next())
      .unwrap()
      .parse()
      .unwrap();
    // Render units are tenths of a point.
    assert!((x / 10.0 - (23.2 - 3.0)).abs() < 0.2, "label ends at {x}");
  }

  #[test]
  fn plot_label_row_draws_inline_graphics_not_placeholder_text() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Row[{Graphics[Disk[]], \" done \", 5}]]",
    );
    assert!(!svg.contains("-Graphics-"), "placeholder leaked");
    assert!(svg.contains("<svg x="), "no nested picture");
    assert!(svg.contains(" done "), "text run missing");
  }

  #[test]
  fn plot_label_padded_row_formats_numbers_beside_graphics() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> PaddedForm[Row[{Graphics[Disk[]], 13.85, \"%\"}], {3, 1}]]",
    );
    assert!(!svg.contains("-Graphics-"));
    assert!(svg.contains(">13.9<") || svg.contains("13.9"), "{svg}");
  }

  #[test]
  fn tall_inline_graphic_in_plot_label_makes_room_above_the_plot() {
    let plain = svg_size(&svg_of("Plot[x, {x, 0, 1}, PlotLabel -> \"a\"]")).1;
    let tall = svg_size(&svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Row[{Graphics[Disk[], ImageSize -> 40], \"a\"}]]",
    ))
    .1;
    assert!(tall > plain, "{tall} vs {plain}");
  }
}
