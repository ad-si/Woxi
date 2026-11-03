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
  // default 1/GoldenRatio shape inside it (WL centres it vertically); it
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
      .filter(|t| t.contains("fill=\"#") && !t.contains("fill=\"none\""))
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
      .split("stroke=\"#FF00")
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

  /// Centres of the `<circle>` data points.
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
    let upper_half = |y: f64| y < (top + bottom) / 2.0;
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
      xs.sort();
      xs.dedup();
      ys.sort();
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
    assert!(!polyline_points(&svg, "#00FF00").is_empty());
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
