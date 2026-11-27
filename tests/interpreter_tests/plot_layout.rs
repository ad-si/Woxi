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

  // The list plots honor a fixed `ImageSize -> {w, h}` the same way:
  // the image keeps both dimensions (it used to lose the height to the
  // default aspect ratio, 450 x 283 instead of 450 x 300).
  #[test]
  fn list_plots_keep_explicit_image_size_pair() {
    for plot in [
      "ListPlot[{1, 4, 2, 5}, ImageSize -> {450, 300}]",
      "ListLinePlot[{1, 4, 2, 5}, ImageSize -> {450, 300}]",
      "ListPlot[{{1, 2}, {3, 5}}, Frame -> True, \
       FrameLabel -> {\"x\", \"y\"}, ImageSize -> {450, 300}]",
      "ListPlot[{1, 4, 2, 5}, ImageSize -> {200, 400}]",
    ] {
      let svg = svg_of(plot);
      let expected = if plot.contains("{200, 400}") {
        (200.0, 400.0)
      } else {
        (450.0, 300.0)
      };
      assert_eq!(svg_size(&svg), expected, "{plot}");
    }
  }

  // A rotated left `FrameLabel` longer than a short fixed-size frame is
  // tall stays inside the image: the padding grows by the label's
  // overhang past the frame's ends, so the frame shrinks (WL: about 83 pt
  // tall in a 213 x 132 image, instead of 94 pt with the label running off
  // the top). `Style[label, 12]` is measured at 12 pt.
  #[test]
  fn long_rotated_frame_label_fits_fixed_size() {
    let svg = svg_of(
      "ListLinePlot[{{0, 0}, {1, 2.5}}, PlotRange -> {{0, 1}, {0, 2.5}}, \
       Frame -> True, FrameLabel -> {{Style[\"force (arbitrary units)\", 12], \
       \"\"}, {Style[\"deformation (arbitrary units)\", 12], \"\"}}, \
       ImageSize -> {213, 132}]",
    );
    assert_eq!(svg_size(&svg), (213.0, 132.0));
    let (_, h) = curve_extent(&svg);
    // The SVG is drawn at 10 render units per point.
    let h = h / 10.0;
    assert!((80.0..88.0).contains(&h), "frame height {h}");
  }

  // A top `FrameLabel` drawn as a picture (a `Row` holding an inline
  // `Graphics`) gets its room above the frame like a text label does,
  // instead of being placed above the image's top edge.
  #[test]
  fn top_picture_frame_label_stays_inside_image() {
    for size in ["ImageSize -> {375, 232}", "ImageSize -> 300"] {
      let svg = svg_of(&format!(
        "ListPlot[{{1, 2, 3}}, Frame -> True, FrameLabel -> {{{{\"l\", \"\"}}, \
         {{\"b\", Row[{{\"key \", Graphics[{{Red, Line[{{{{0, 0}}, {{10, 0}}}}]}}, \
         ImageSize -> {{30, 20}}]}}]}}}}, {size}]"
      ));
      // The label's own `<svg>` is the first one nested in the image.
      let nested = svg.match_indices("<svg ").nth(1).map(|(i, _)| i).unwrap();
      let tag = &svg[nested..nested + svg[nested..].find('>').unwrap()];
      let start = tag.find(" y=\"").unwrap() + 4;
      let y: f64 = tag[start..start + tag[start..].find('"').unwrap()]
        .parse()
        .unwrap();
      assert!(y >= 0.0, "{size}: label starts at y = {y}: {tag}");
    }
  }

  // The text of such a picture label is set like a text frame label, in
  // gray sans-serif, while a `Style` around the label (its size) and around
  // an item (italic, a color) still applies.
  #[test]
  fn picture_frame_label_text_takes_frame_label_style() {
    for edge in ["{\"b\", LABEL}", "{LABEL, \"\"}"] {
      let label = "Style[Row[{\"key \", Style[\"t\", Italic], \" \", \
         Style[\"hot\", Red], Graphics[{Red, Line[{{0, 0}, {10, 0}}]}, \
         ImageSize -> {30, 20}]}], 17]";
      let svg = svg_of(&format!(
        "ListPlot[{{1, 2, 3}}, Frame -> True, FrameLabel -> {{{{\"l\", \"\"}}, \
         {}}}]",
        edge.replace("LABEL", label)
      ));
      let text_tag = |content: &str| {
        let end = svg.find(&format!(">{content}")).unwrap();
        &svg[svg[..end].rfind("<text").unwrap()..end]
      };
      let key = text_tag("key ");
      assert!(key.contains("sans-serif"), "{edge}: {key}");
      assert!(key.contains("fill=\"rgb(102,102,102)\""), "{edge}: {key}");
      assert!(key.contains("font-size=\"17\""), "{edge}: {key}");
      let hot = text_tag("hot");
      assert!(hot.contains("fill=\"rgb(255,0,0)\""), "{edge}: {hot}");
      let italic = svg.match_indices("\">t<").any(|(end, _)| {
        let start = svg[..end].rfind("<text").unwrap();
        svg[start..end].contains("font-style=\"italic\"")
      });
      assert!(italic, "{edge}");
    }
  }

  // `ImageSize -> {Automatic, h}` fixes the height of a list plot too.
  #[test]
  fn list_plot_fixed_height() {
    let svg = svg_of("ListPlot[{1, 4, 2, 5}, ImageSize -> {Automatic, 300}]");
    assert_eq!(svg_size(&svg).1, 300.0);
  }

  // A bare width still derives the height from the aspect ratio.
  #[test]
  fn width_only_still_follows_aspect_ratio() {
    let svg = svg_of("Plot[x, {x, 0, 1}, ImageSize -> 300, AspectRatio -> 1]");
    let (width, height) = svg_size(&svg);
    assert_eq!(width, 300.0);
    assert!(height > 250.0, "height {height}");
  }

  /// The `font-size` of the horizontal axis label `<text …>{label}<`.
  fn x_axis_label_font(svg: &str, label: &str) -> f64 {
    let end = svg.find(&format!(">{label}</text>")).unwrap();
    let tag = &svg[svg[..end].rfind("<text").unwrap()..end];
    let start = tag.find("font-size=\"").unwrap() + 11;
    let stop = start + tag[start..].find('"').unwrap();
    tag[start..stop].parse().unwrap()
  }

  // A narrow axis label fits the room the automatic padding reserved for
  // it, so a square plot in a fixed image keeps it at the tick label size
  // (it used to shrink when the plot was sized by the image height).
  #[test]
  fn narrow_axis_label_keeps_its_size_in_square_plot() {
    for y_max in ["1700", "60", "10"] {
      let svg = svg_of(&format!(
        "Plot[x, {{x, 0, 15}}, AspectRatio -> 1, AxesLabel -> {{t, v}}, \
         PlotRange -> {{{{0, 30}}, {{0, {y_max}}}}}, ImageSize -> {{195, 200}}]"
      ));
      assert_eq!(
        x_axis_label_font(&svg, "<tspan font-style=\"italic\">t</tspan>"),
        100.0,
        "PlotRange max {y_max}"
      );
    }
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

  // `PolarGridLines -> Automatic` draws, under the points, 25 spokes every
  // Pi/12 out to the farthest point and circles at fifths of that radius
  // (Wolfram's own grid for the axes-free plot), and the plot range spans
  // the whole grid: the spokes at angles 0 and Pi reach both edges.
  #[test]
  fn polar_grid_lines_automatic() {
    let svg = svg_of(
      "ListPolarPlot[{{0.5, 1}, {2., -1.5}, {Pi/2, 2}}, \
       PolarGridLines -> Automatic, Axes -> False, ImageSize -> {200, 200}]",
    );
    assert_eq!(svg.matches("<polyline").count(), 25, "{svg}");
    let radii: Vec<f64> = svg
      .split("<ellipse ")
      .skip(1)
      .map(|tag| {
        let start = tag.find("rx=\"").unwrap() + 4;
        let end = start + tag[start..].find('"').unwrap();
        tag[start..end].parse().unwrap()
      })
      .collect();
    assert_eq!(radii.len(), 5, "{svg}");
    // The first spoke runs from the center along angle 0 to radius 2.
    let spoke = svg.split("<polyline").nth(1).unwrap();
    let start = spoke.find("points=\"").unwrap() + 8;
    let end = start + spoke[start..].find('"').unwrap();
    let coords: Vec<f64> = spoke[start..end]
      .split([' ', ','])
      .map(|c| c.parse().unwrap())
      .collect();
    let (cx, cy, x1, y1) = (coords[0], coords[1], coords[2], coords[3]);
    assert!((cy - y1).abs() < 0.5, "{coords:?}");
    let r_max = x1 - cx;
    for (k, r) in radii.iter().enumerate() {
      assert!(
        (r - r_max * k as f64 / 5.0).abs() < 0.5,
        "{radii:?} {r_max}"
      );
    }
    // The plot range takes in the spokes at angles 0 and Pi.
    assert!(cx - r_max > 0.0 && x1 < 2000.0, "{coords:?}");
    assert!((2.0 * cx - 2000.0).abs() < 1.0, "{coords:?}");
    // No grid unless asked for.
    let plain =
      svg_of("ListPolarPlot[{{0.5, 1}, {2., -1.5}, {Pi/2, 2}}, Axes -> False]");
    assert!(!plain.contains("<polyline"), "{plain}");
    assert!(!plain.contains("<ellipse"), "{plain}");
  }

  // `PolarGridLines -> {None, Automatic}` keeps only the circles, and
  // explicit angle lists draw one spoke per angle.
  #[test]
  fn polar_grid_lines_parts() {
    let polylines = |svg: &str| svg.matches("<polyline").count();
    let radial = svg_of(
      "ListPolarPlot[{{0, 1}, {Pi/2, 2}}, \
       PolarGridLines -> {None, Automatic}, Axes -> False]",
    );
    assert_eq!(polylines(&radial), 0, "{radial}");
    assert_eq!(radial.matches("<ellipse").count(), 5, "{radial}");
    let spokes = svg_of(
      "ListPolarPlot[{{0, 1}, {Pi/2, 2}}, \
       PolarGridLines -> {{0, Pi/2, Pi}, None}, Axes -> False]",
    );
    assert_eq!(polylines(&spokes), 3, "{spokes}");
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

  /// `Show` keeps the `PlotLabels` of every plot it combines, each on its
  /// own curves. Regression: only the first plot's labels survived, so a
  /// `ListPlot`'s label shown over a `Plot` was dropped.
  #[test]
  fn show_keeps_the_labels_of_every_plot() {
    let points = "{{0.2, 0.3}, {0.5, 0.6}}";
    for labels in ["{Placed[\"pts\", After]}", "{\"pts\"}", "\"pts\""] {
      let svg = svg_of(&format!(
        "Show[{{Plot[x, {{x, 0, 1}}], ListPlot[{points}, PlotLabels -> {labels}]}}]"
      ));
      assert!(svg.contains(">pts</text>"), "{labels}: {svg}");
    }
    // Two labeled plots of curves label all three curves, as one plot of
    // the three does.
    let shown = svg_of(
      "Show[Plot[{x, x^2}, {x, 0, 1}, PlotLabels -> {\"a\", \"b\"}], \
       Plot[x^3, {x, 0, 1}, PlotLabels -> {\"c\"}]]",
    );
    let alone = svg_of(
      "Plot[{x, x^2, x^3}, {x, 0, 1}, PlotLabels -> {\"a\", \"b\", \"c\"}]",
    );
    for l in ["a", "b", "c"] {
      assert_eq!(label_at(&shown, l), label_at(&alone, l), "{l}");
    }
    // Unlabeled later plots leave the first plot's labels as they were.
    let shown = svg_of(
      "Show[Plot[{x, x^2}, {x, 0, 1}, PlotLabels -> {\"a\", \"b\"}], \
       Plot[x^3, {x, 0, 1}]]",
    );
    assert!(shown.contains(">a</text>") && shown.contains(">b</text>"));
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

  /// `ImagePadding -> {{l, r}, {Automatic, Automatic}}` fixes the sides it
  /// gives and leaves the others to the decorations, instead of being
  /// dropped whole.
  #[test]
  fn image_padding_with_automatic_sides_fixes_the_given_ones() {
    let tick_x = |svg: &str, label: &str| -> f64 {
      let end = svg.find(&format!(">{label}</text>")).expect("tick label");
      let tag = &svg[svg[..end].rfind("<text").unwrap()..end];
      let start = tag.find(" x=\"").unwrap() + 4;
      tag[start..start + tag[start..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    let plot = |options: &str| {
      svg_of(&format!(
        "Plot[3, {{x, 0, 100}}, PlotRange -> {{{{0, 100}}, {{0, 20}}}},          ImageSize -> 300{options}]"
      ))
    };
    // 50 pt on both sides of a 300 pt image: the x axis spans 50..250.
    let both = plot(", ImagePadding -> {{50, 50}, {Automatic, Automatic}}");
    assert_eq!(tick_x(&both, "0"), 500.0);
    assert_eq!(tick_x(&both, "100"), 2500.0);
    // The automatic sides are the ones the plot has without the option.
    let auto = plot("");
    let left = plot(", ImagePadding -> {{50, Automatic}, {Automatic, 40}}");
    assert_eq!(tick_x(&left, "0"), 500.0);
    assert_eq!(tick_x(&left, "100"), tick_x(&auto, "100"));
    let right = plot(", ImagePadding -> {{All, 80}, {Automatic, Automatic}}");
    assert_eq!(tick_x(&right, "0"), tick_x(&auto, "0"));
    assert_eq!(tick_x(&right, "100"), 2200.0);
    // The same holds for a plot of data.
    let data = "ListLinePlot[{{0, 1}, {100, 2}}, PlotRange -> {{0, 100}, \
                {0, 20}}, ImageSize -> 300, ImagePadding -> {{50, 50}, \
                {Automatic, 10}}]";
    let svg = svg_of(data);
    assert_eq!(tick_x(&svg, "0"), 500.0);
    assert_eq!(tick_x(&svg, "100"), 2500.0);
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

/// A curve whose values are all far below 1 still gets a y range of its
/// own instead of being taken for a constant and flattened onto 0.
mod tiny_values {
  use super::*;

  fn tick_labels(svg: &str) -> Vec<String> {
    svg
      .split("<text ")
      .skip(1)
      .map(|t| {
        let start = t.find('>').unwrap() + 1;
        let end = start + t[start..].find("</text>").unwrap();
        t[start..end].to_string()
      })
      .collect()
  }

  #[test]
  fn varying_curve_fills_the_plot() {
    let svg = svg_of("Plot[10^-20 x, {x, 1, 2}, ImageSize -> 300]");
    let (w, h) = curve_extent(&svg);
    assert!(w > 2000.0, "curve width {w}");
    assert!(h > 1000.0, "curve height {h}");
    let labels = tick_labels(&svg);
    assert!(labels.iter().any(|l| l == "2.0×10⁻²⁰"), "{labels:?}");
    assert!(
      !labels.iter().any(|l| l == "-1.0" || l == "1.0"),
      "{labels:?}"
    );
  }

  #[test]
  fn tiny_exponential_curve_keeps_its_scale() {
    let svg = svg_of("Plot[Exp[-100000/t], {t, 2600, 2700}, ImageSize -> 300]");
    let (_, h) = curve_extent(&svg);
    assert!(h > 1000.0, "curve height {h}");
    let labels = tick_labels(&svg);
    assert!(labels.iter().any(|l| l.contains("10⁻¹⁷")), "{labels:?}");
  }

  /// Every scientific label of an axis shows as many mantissa decimals as
  /// the one needing the most, as WL labels a tick set: an axis stepping
  /// by `5*10^-8` reads `5.0×10⁻⁸, 1.0×10⁻⁷, 1.5×10⁻⁷, 2.0×10⁻⁷`.
  #[test]
  fn scientific_tick_mantissas_share_their_decimals() {
    for (code, expected) in [
      (
        "Plot[x/10^7, {x, 0, 2}, Axes -> {False, True}]",
        ["0", "5.0×10⁻⁸", "1.0×10⁻⁷", "1.5×10⁻⁷", "2.0×10⁻⁷"],
      ),
      (
        "Plot[x, {x, 0, 2*10^15}, Axes -> {True, False}]",
        ["0", "5.0×10¹⁴", "1.0×10¹⁵", "1.5×10¹⁵", "2.0×10¹⁵"],
      ),
      (
        "Plot[x/10^7, {x, 0, 2}, Frame -> True, Axes -> False, \
         FrameTicks -> {{Automatic, None}, {None, None}}]",
        ["0", "5.0×10⁻⁸", "1.0×10⁻⁷", "1.5×10⁻⁷", "2.0×10⁻⁷"],
      ),
    ] {
      let labels: Vec<String> = tick_labels(&svg_of(code))
        .into_iter()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
      assert_eq!(labels, expected, "{code}");
    }
    // Integer mantissas alone stay bare.
    let labels =
      tick_labels(&svg_of("Plot[x, {x, 0, 10^7}, Axes -> {True, False}]"));
    assert!(labels.iter().any(|l| l == "2×10⁶"), "{labels:?}");
    assert!(labels.iter().any(|l| l == "1×10⁷"), "{labels:?}");
  }

  #[test]
  fn tiny_constant_spans_zero_to_twice_its_value() {
    let svg = svg_of("Plot[10^-20, {x, 0, 1}]");
    let labels = tick_labels(&svg);
    assert!(labels.iter().any(|l| l == "1.0×10⁻²⁰"), "{labels:?}");
    assert!(labels.iter().any(|l| l == "2.0×10⁻²⁰"), "{labels:?}");
  }
}

mod labeled_frame_label {
  use super::*;

  /// The text of every `<text>` element (and its `<tspan>` lines), in order.
  fn text_lines(svg: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in svg.split("<text ").skip(1) {
      let body =
        &chunk[chunk.find('>').unwrap() + 1..chunk.find("</text>").unwrap()];
      for part in body.split('<') {
        let text = part.rsplit('>').next().unwrap_or(part).trim();
        if !text.is_empty() && !text.starts_with('/') {
          out.push(text.to_string());
        }
      }
    }
    out
  }

  #[test]
  fn labeled_stacks_its_label_below_the_content() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {\"a\", \"b\", \
       Labeled[NumberForm[88.5321, {5, 1}], \"mileage\"], \"\"}]",
    );
    let lines = text_lines(&svg);
    assert!(!lines.iter().any(|l| l.contains("Labeled")), "{lines:?}");
    let value = lines.iter().position(|l| l == "88.5").expect("value");
    assert_eq!(lines.get(value + 1).map(String::as_str), Some("mileage"));
  }

  #[test]
  fn labeled_top_stacks_its_label_above_the_content() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, Frame -> True, \
       FrameLabel -> {Labeled[\"t\", \"unit\", Top], \"y\"}]",
    );
    let lines = text_lines(&svg);
    let label = lines.iter().position(|l| l == "unit").expect("label");
    assert_eq!(lines.get(label + 1).map(String::as_str), Some("t"));
  }

  #[test]
  fn labeled_axes_label_stacks_too() {
    let svg =
      svg_of("Plot[x, {x, 0, 1}, AxesLabel -> {Labeled[\"t\", \"s\"], \"y\"}]");
    let lines = text_lines(&svg);
    assert!(!lines.iter().any(|l| l.contains("Labeled")), "{lines:?}");
    let content = lines.iter().position(|l| l == "t").expect("content");
    assert_eq!(lines.get(content + 1).map(String::as_str), Some("s"));
  }
}

mod base_style {
  use super::*;

  fn font_sizes(svg: &str) -> Vec<f64> {
    svg
      .split("font-size=\"")
      .skip(1)
      .map(|s| s[..s.find('"').unwrap()].parse::<f64>().unwrap())
      .collect()
  }

  #[test]
  fn base_style_font_size_reaches_every_label() {
    for spec in ["12", "{12}", "{FontSize -> 12}"] {
      let svg = svg_of(&format!(
        "Plot[x, {{x, 0, 1}}, Frame -> True, FrameLabel -> {{\"a\", \"b\"}}, \
         BaseStyle -> {spec}]"
      ));
      let sizes = font_sizes(&svg);
      assert_ne!(sizes, [] as [f64; 0]);
      assert!(
        sizes.iter().all(|s| (s - 120.0).abs() < 1e-6),
        "{spec}: {sizes:?}"
      );
    }
  }

  #[test]
  fn label_style_overrides_base_style_whichever_comes_first() {
    for opts in [
      "BaseStyle -> 12, LabelStyle -> 14",
      "LabelStyle -> 14, BaseStyle -> 12",
    ] {
      let svg = svg_of(&format!(
        "Plot[x, {{x, 0, 1}}, Frame -> True, FrameLabel -> {{\"a\"}}, {opts}]"
      ));
      let sizes = font_sizes(&svg);
      assert!(
        sizes.iter().all(|s| (s - 140.0).abs() < 1e-6),
        "{opts}: {sizes:?}"
      );
    }
  }
}

mod base_style_labels {
  use super::*;

  /// The opening `<text …>` tag of each text element whose content starts
  /// with `content`.
  fn text_tags<'a>(svg: &'a str, content: &str) -> Vec<&'a str> {
    svg
      .match_indices("<text ")
      .filter_map(|(i, _)| {
        let end = i + svg[i..].find('>')?;
        let rest = &svg[end + 1..];
        let body = rest
          .strip_prefix("<tspan font-size=\"18\">")
          .unwrap_or(rest);
        body.starts_with(content).then_some(&svg[i..=end])
      })
      .collect()
  }

  fn attr(tag: &str, name: &str) -> Option<f64> {
    let start = tag.find(&format!(" {name}=\""))? + name.len() + 3;
    tag[start..start + tag[start..].find('"')?].parse().ok()
  }

  // The list plots read `BaseStyle` like `Plot` does: its size reaches
  // the frame labels and tick labels.
  #[test]
  fn list_plots_honor_base_style_font_size() {
    for head in ["ListPlot", "ListLinePlot"] {
      let svg = svg_of(&format!(
        "{head}[{{1, 2, 1.5}}, Frame -> True, FrameLabel -> {{\"a\", \"b\"}}, \
         BaseStyle -> {{18}}]"
      ));
      let tags = text_tags(&svg, "a");
      assert_eq!(tags.len(), 1, "{head}: {svg}");
      assert_eq!(attr(tags[0], "font-size"), Some(180.0), "{head}");
    }
  }

  // `BaseStyle -> {…, Italic}` (or `LabelStyle -> Italic`) slants the
  // frame labels and the title of a plot.
  #[test]
  fn italic_base_style_slants_plot_labels() {
    for code in [
      "Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {\"a\", \"b\"}, \
       PlotLabel -> \"T\", BaseStyle -> {14, Italic}]",
      "ListLinePlot[{1, 2, 1.5}, Frame -> True, \
       FrameLabel -> {\"a\", \"b\"}, PlotLabel -> \"T\", \
       LabelStyle -> Italic]",
    ] {
      let svg = svg_of(code);
      for label in ["a", "b", "T"] {
        let tags = text_tags(&svg, label);
        assert_eq!(tags.len(), 1, "{code}: {label}");
        assert!(tags[0].contains("font-style=\"italic\""), "{code}: {label}");
      }
    }
  }

  // A `Graphics` (as `Show` of a plot returns) sets its frame labels and
  // title in the `BaseStyle` size and slant.
  #[test]
  fn graphics_frame_labels_follow_base_style() {
    for code in [
      "Graphics[Line[{{0, 0}, {1, 1}}], Frame -> True, \
       FrameLabel -> {\"a\", \"b\"}, PlotLabel -> \"T\", \
       BaseStyle -> {18, Italic}]",
      "Show[ListLinePlot[{1, 2, 1.5}, Frame -> True, \
       FrameLabel -> {\"a\", \"b\"}, PlotLabel -> \"T\", \
       BaseStyle -> {18, Italic}], Graphics[Line[{{1, 1}, {2, 1}}]]]",
    ] {
      let svg = svg_of(code);
      for label in ["a", "b", "T"] {
        let tags = text_tags(&svg, label);
        assert_eq!(tags.len(), 1, "{code}: {label}");
        assert_eq!(attr(tags[0], "font-size"), Some(18.0), "{code}: {label}");
        assert!(tags[0].contains("font-style=\"italic\""), "{code}: {label}");
      }
    }
  }

  // A title piece styled larger than the default title font
  // (`Style["T", 18]`) gets a band tall enough for it: its baseline sits
  // a full 18-point ascent below the top instead of being cut off.
  #[test]
  fn graphics_title_band_fits_a_larger_styled_title() {
    let svg = svg_of(
      "Show[ListLinePlot[{1, 2, 1.5}, PlotLabel -> Style[\"T\", 18]], \
       Graphics[Line[{{1, 1}, {2, 1}}]]]",
    );
    let tags = text_tags(&svg, "T");
    assert_eq!(tags.len(), 1, "{svg}");
    let y = attr(tags[0], "y").unwrap();
    assert!(y >= 18.0 * 1854.0 / 2048.0 - 0.01, "{y}");
  }
}

mod clipped_plot_range_sampling {
  use super::*;

  /// How many points of the default-colored curve are drawn.
  fn curve_points(svg: &str) -> usize {
    polyline_points(svg, "#3D99CC").len()
  }

  // A curve whose samples span far more than an explicit `PlotRange` must
  // still be refined within the visible window: a cubic on {x, -10, 10}
  // clipped to {-10, 10} draws as a smooth curve, not a polygon of the
  // dozen initial samples that fall inside the window.
  #[test]
  fn cubic_refined_inside_clipped_range() {
    let svg =
      svg_of("Plot[x^3 - 2 x^2 - 3 x, {x, -10, 10}, PlotRange -> {-10, 10}]");
    let n = curve_points(&svg);
    assert!(n >= 35, "only {n} curve points: {svg}");
  }

  #[test]
  fn quadratic_refined_inside_clipped_range() {
    let svg = svg_of("Plot[x^2 + 5, {x, -20, 20}, PlotRange -> {0, 10}]");
    let n = curve_points(&svg);
    assert!(n >= 35, "only {n} curve points: {svg}");
  }

  // Samples that overshoot the PlotRange edge only by rounding noise (a
  // running total of floats that should end exactly at the edge) lie on
  // the edge: the curve keeps running along it instead of being cut off.
  #[test]
  fn rounding_overshoot_stays_on_edge() {
    let svg = svg_of(
      "ListPlot[{0, 50, 100.00000000000003, 100.00000000000003, \
       100.00000000000003}, PlotRange -> {0, 100}, Joined -> True]",
    );
    assert_eq!(curve_points(&svg), 5, "{svg}");
    let svg = svg_of(
      "ListPlot[{0, 0.5, 1.0000000000000002, 1.0000000000000002}, \
       PlotRange -> {0, 1}, Joined -> True]",
    );
    assert_eq!(curve_points(&svg), 4, "{svg}");
  }

  // A real overshoot is still clipped at the edge.
  #[test]
  fn real_overshoot_still_clipped() {
    let svg = svg_of(
      "ListPlot[{0, 50, 120, 120, 50}, PlotRange -> {0, 100}, Joined -> True]",
    );
    let polylines = svg.matches("stroke=\"#3D99CC\"").count();
    assert_eq!(polylines, 2, "{svg}");
  }
}

mod traditional_plot_label {
  use super::*;

  /// The SVG `<text>` element holding the plot title.
  fn title(svg: &str) -> String {
    let start = svg.rfind("<text ").unwrap();
    let end = start + svg[start..].find("</text>").unwrap();
    let tag = &svg[start..end];
    // Strip the markup, keep the visible text.
    let mut out = String::new();
    let mut in_tag = false;
    for c in tag.chars() {
      match c {
        '<' => in_tag = true,
        '>' => in_tag = false,
        _ if !in_tag => out.push(c),
        _ => {}
      }
    }
    out
  }

  // A graphic's `FormatType` is `TraditionalForm`, so a polynomial in the
  // label runs from its highest power down.
  #[test]
  fn polynomial_label_descending() {
    let svg = svg_of("Plot[x, {x, 0, 1}, PlotLabel -> x^3 - 2 x^2 - 3 x]");
    assert_eq!(title(&svg), "x3-2\u{2009}x2-3\u{2009}x");
  }

  #[test]
  fn polynomial_inside_row_label_descending() {
    let svg =
      svg_of("Plot[x, {x, 0, 1}, PlotLabel -> Row[{\"g = \", 1 + y + y^2}]]");
    assert_eq!(title(&svg), "g = y2+y+1");
  }

  #[test]
  fn polynomial_inside_styled_row_label_descending() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Style[Row[{\"g = \", x + x^2}], Bold]]",
    );
    assert_eq!(title(&svg), "g = x2+x");
  }
}

mod typeset_labels {
  use super::*;

  // A label expression that typesets as a fraction is laid out as a
  // stacked picture inline with the rest of the label, not as a multi-line
  // text block of `-`-ruled rows.
  #[test]
  fn text_fraction_label_stacks_inline() {
    let svg = svg_of(
      "Graphics[Text[Row[{-x^2, Row[{\"d\", \"t\"}]/Row[{\"d\", \"x\"}], \
       \" = \", 1.5}], {0, 0}]]",
    );
    assert!(!svg.contains("---"), "{svg}");
    assert!(!svg.contains("dy=\"14\""), "{svg}");
    // The fraction is drawn with a rule between numerator and denominator.
    assert!(svg.contains("<line") || svg.contains("<rect"), "{svg}");
    assert!(svg.contains("1.5"), "{svg}");
  }

  // The stacked label writes `Pi` as `π`, and a string part (the unit
  // `"A"`) stays upright instead of being set as an italic variable.
  #[test]
  fn text_fraction_label_pi_glyph_and_upright_strings() {
    let svg = svg_of("Graphics[Text[Row[{\"I = \", 7 Pi/10, \"A\"}], {0, 0}]]");
    assert!(svg.contains(">π</text>"), "{svg}");
    assert!(!svg.contains(">Pi</text>"), "{svg}");
    assert!(
      svg.contains("stroke=\"none\" xml:space=\"preserve\">A</text>"),
      "{svg}"
    );
    assert!(!svg.contains("font-style=\"italic\" xml:space=\"preserve\">A<"));
  }

  // A `TraditionalForm` label raises its exponents instead of flattening
  // `x^n` to the run `xn`, and keeps the size of its `Style` wrapper.
  #[test]
  fn text_traditional_form_power_label_raises_exponent() {
    let svg = svg_of("Graphics[Text[Style[TraditionalForm[x^n], 16], {0, 0}]]");
    assert!(!svg.contains(">xn</text>"), "{svg}");
    assert!(svg.contains("<svg x="), "{svg}");
    assert!(
      svg.contains("font-size=\"16.0\" stroke=\"none\" font-style=\"italic\" xml:space=\"preserve\">x</text>"),
      "{svg}"
    );
    assert!(svg.contains("font-size=\"11.2\""), "{svg}");
    let svg = svg_of("Graphics[Text[TraditionalForm[4^n (a b)^n], {0, 0}]]");
    assert!(!svg.contains(">4n"), "{svg}");
    assert!(svg.contains("<svg x="), "{svg}");
  }

  // A `TraditionalForm` quotient stacks as a fraction.
  #[test]
  fn text_traditional_form_quotient_label_stacks() {
    let svg = svg_of("Graphics[Text[TraditionalForm[x/y], {0, 0}]]");
    assert!(!svg.contains(">xy</text>"), "{svg}");
    assert!(svg.contains("<svg x="), "{svg}");
  }

  #[test]
  fn text_quotient_label_stacks() {
    let svg = svg_of("Graphics[Text[a^2/b, {0, 0}]]");
    assert!(!svg.contains("dy=\"14\""), "{svg}");
    assert!(svg.contains("<svg x="), "{svg}");
  }

  // `Text[Style[label, size]]` as a frame label keeps the style's size.
  #[test]
  fn frame_label_text_style_keeps_size() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {Text[Style[\"a\", 14]], \"b\"}]",
    );
    assert!(svg.contains("<tspan font-size=\"140\">a</tspan>"), "{svg}");
  }

  // A styled frame label still sets its expression in traditional notation.
  #[test]
  fn styled_frame_label_is_traditional() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {\"x\", Style[f[x], 14]}]",
    );
    assert!(svg.contains(">f</tspan>(<tspan"), "{svg}");
    assert!(!svg.contains("f[x]"), "{svg}");
  }

  // Re-rendering a plot through `Show` measures a styled top frame label by
  // its visible text, not its markup, so the frame keeps its margins.
  #[test]
  fn show_styled_top_frame_label_keeps_margins() {
    let plain = svg_of(
      "Show[Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {{\"a\", \" \"}, \
       {\"x\", \"L n = 5\"}}, ImageSize -> {500, 350}], Graphics[Point[{0, 0}]]]",
    );
    let styled = svg_of(
      "Show[Plot[x, {x, 0, 1}, Frame -> True, FrameLabel -> {{\"a\", \" \"}, \
       {\"x\", Style[Row[{\"L \", Style[\"n\", Italic], \" = \", 5}], 16]}}, \
       ImageSize -> {500, 350}], Graphics[Point[{0, 0}]]]",
    );
    let translate = |svg: &str| {
      let start = svg.find("translate(").unwrap() + "translate(".len();
      let end = start + svg[start..].find(',').unwrap();
      svg[start..end].parse::<f64>().unwrap()
    };
    assert!((translate(&plain) - translate(&styled)).abs() < 1.0);
  }
}

/// Regression: a per-series `PlotStyle` entry that is a directive list
/// holding a color (`{Opacity[a], Gray}`) was read as that one color, so
/// the curve lost its opacity (and any other directive in the list).
mod plot_style_directive_lists {
  use super::*;

  /// The `opacity` of every curve (`<polyline>`) in order.
  fn curve_opacities(svg: &str) -> Vec<String> {
    svg
      .split("<polyline ")
      .skip(1)
      .map(|t| {
        let start = t.find("opacity=\"").unwrap() + "opacity=\"".len();
        t[start..start + t[start..].find('"').unwrap()].to_string()
      })
      .collect()
  }

  #[test]
  fn plot_keeps_opacity_beside_color() {
    let svg = svg_of(
      "Plot[{Sin[x], Cos[x]}, {x, 0, 1}, \
       PlotStyle -> {{Opacity[0.3], Gray}, {Red, Opacity[0.5]}}]",
    );
    assert_eq!(curve_opacities(&svg), ["0.3", "0.5"], "{svg}");
    assert!(svg.contains("stroke=\"#808080\""), "{svg}");
    assert!(svg.contains("stroke=\"#FF0000\""), "{svg}");
  }

  #[test]
  fn parametric_plot_keeps_opacity_beside_color() {
    let svg = svg_of(
      "ParametricPlot[{{Cos[t], Sin[t]}, {2 Cos[t], Sin[t]}}, {t, 0, 2 Pi}, \
       PlotStyle -> {{Opacity[0.025], Gray}, {Opacity[0.025], Gray}}]",
    );
    assert_eq!(curve_opacities(&svg), ["0.025", "0.025"], "{svg}");
  }

  #[test]
  fn single_series_list_keeps_opacity() {
    let svg =
      svg_of("Plot[Sin[x], {x, 0, 1}, PlotStyle -> {Red, Opacity[0.5]}]");
    assert_eq!(curve_opacities(&svg), ["0.5"], "{svg}");
  }
}

mod label_directive_lists {
  use super::*;

  /// The `<text …>` opening tag of the element that draws `label`.
  fn text_tag(svg: &str, label: &str) -> String {
    let end = svg.find(&format!(">{label}")).unwrap();
    let start = svg[..end].rfind("<text").unwrap();
    svg[start..end].to_string()
  }

  #[test]
  fn style_directive_list_applies_each_directive() {
    let listed =
      svg_of("Plot[x, {x, 0, 1}, PlotLabel -> Style[\"lab\", {Orange, 18}]]");
    let spread =
      svg_of("Plot[x, {x, 0, 1}, PlotLabel -> Style[\"lab\", Orange, 18]]");
    let tag = text_tag(&listed, "lab");
    assert!(tag.contains("fill=\"rgb(255,128,0)\""), "{tag}");
    assert_eq!(tag, text_tag(&spread, "lab"));
  }

  #[test]
  fn nested_style_directive_lists_flatten() {
    let nested = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Style[\"lab\", {{Bold, Red}, 14}]]",
    );
    let spread =
      svg_of("Plot[x, {x, 0, 1}, PlotLabel -> Style[\"lab\", Bold, Red, 14]]");
    assert_eq!(text_tag(&nested, "lab"), text_tag(&spread, "lab"));
  }
}

mod tiny_scale_sampling {
  use super::*;

  /// Adaptive refinement does not depend on the curve's absolute scale: a
  /// peak of height `10^-12` is sampled as finely as the same peak of
  /// height 1, instead of staying a coarse zigzag.
  #[test]
  fn tiny_curve_refines_like_unit_curve() {
    let unit = svg_of("Plot[1/(1 + 100 x^2), {x, -1, 1}]");
    let tiny = svg_of("Plot[10^-12/(1 + 100 x^2), {x, -1, 1}]");
    let unit_n = polyline_points(&unit, "#3D99CC").len();
    let tiny_n = polyline_points(&tiny, "#3D99CC").len();
    assert!(unit_n > 30, "{unit_n}");
    assert_eq!(unit_n, tiny_n);
  }
}

mod scientific_tick_labels {
  use super::*;

  #[test]
  fn mantissas_share_decimals_across_decades() {
    for opts in ["", ", Frame -> True"] {
      let svg = svg_of(&format!("Plot[x, {{x, 0, 3*10^16}}{opts}]"));
      for label in ["5.0×10¹⁵", "1.0×10¹⁶", "1.5×10¹⁶", "3.0×10¹⁶"]
      {
        assert!(svg.contains(label), "missing {label} ({opts})");
      }
      assert!(!svg.contains(">5×10¹⁵"), "{opts}");
    }
  }

  #[test]
  fn integer_mantissas_stay_integers() {
    let svg = svg_of("Plot[x, {x, 0, 10^7}]");
    for label in ["2×10⁶", "8×10⁶", "1×10⁷"] {
      assert!(svg.contains(label), "missing {label}");
    }
    let svg = svg_of("Plot[x, {x, 0, 6*10^-12}, Frame -> True]");
    assert!(svg.contains("1×10⁻¹²") && svg.contains("6×10⁻¹²"));
  }
}

mod column_plot_label {
  use super::*;

  /// The `<text …>` element of the title, from its opening tag up to the
  /// text that starts its first line.
  fn title_tag<'a>(svg: &'a str, first_line: &str) -> &'a str {
    let end = svg.find(first_line).expect("title line");
    let start = svg[..end].rfind("<text").expect("title element");
    &svg[start..end]
  }

  fn attr(tag: &str, name: &str) -> String {
    let start = tag.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
    let end = start + tag[start..].find('"').unwrap();
    tag[start..end].to_string()
  }

  #[test]
  fn column_lines_align_left_in_a_centered_block() {
    // A `Column` sets its rows flush left (its default `Alignment` is
    // `Left`); the block as a whole stays centered over the plot.
    for plot in ["ListPlot[{1, 2, 3}", "Graphics[{Circle[]}"] {
      let svg = svg_of(&format!(
        "{plot}, PlotLabel -> Column[{{\"a long first line\", \"short\"}}]]"
      ));
      let tag = title_tag(&svg, "a long first line");
      assert_eq!(attr(tag, "text-anchor"), "start", "{plot}: {svg}");
      let x = attr(tag, "x");
      assert!(
        svg.contains(&format!("<tspan x=\"{x}\" dy=")),
        "{plot}: every line starts at x = {x}: {svg}"
      );
    }
  }

  #[test]
  fn centered_column_and_grid_center_each_line() {
    for label in [
      "Column[{\"a long first line\", \"short\"}, Center]",
      "Column[{\"a long first line\", \"short\"}, Alignment -> Center]",
      "Grid[{{\"a long first line\"}, {\"short\"}}]",
    ] {
      for plot in ["ListPlot[{1, 2, 3}", "Graphics[{Circle[]}"] {
        let svg = svg_of(&format!("{plot}, PlotLabel -> {label}]"));
        let tag = title_tag(&svg, "a long first line");
        assert_eq!(attr(tag, "text-anchor"), "middle", "{label}: {svg}");
      }
    }
  }

  #[test]
  fn label_style_sets_a_stacked_title_small() {
    // `Style[Column[…], "Label"]` takes the 9-point size of the "Label"
    // stylesheet entry, not the default 12-point title size.
    let svg = svg_of(
      "ListPlot[{1, 2, 3}, PlotLabel -> Style[Column[{\"top\", \"bottom\"}], \"Label\"]]",
    );
    let tag = title_tag(&svg, "top");
    assert_eq!(attr(tag, "font-size"), "90", "{svg}");
    // An explicit size wins over the named style's, wherever it stands.
    let svg = svg_of(
      "ListPlot[{1, 2, 3}, PlotLabel -> Style[Column[{\"top\", \"bottom\"}], 20, \"Label\"]]",
    );
    let tag = title_tag(&svg, "top");
    assert_eq!(attr(tag, "font-size"), "200", "{svg}");
  }

  // Where one factor of the plotted function underflows and another leaves
  // the machine range, the function is `0.` and the curve carries on to the
  // end of the range instead of breaking off at an `Indeterminate`.
  #[test]
  fn curve_continues_where_a_factor_overflows() {
    assert_eq!(
      interpret(
        "Max[Cases[Plot[E^(-E^x) (E^(E^x) - 1), {x, 0, 20}], \
         Line[p_] :> p[[All, 1]], Infinity]] > 19.9"
      )
      .unwrap(),
      "True"
    );
  }
}

/// `Show` draws the axes and automatic grid lines of a framed plot the way
/// the plot does on its own: only strictly inside the plot's unpadded range.
mod show_framed_plot_axes {
  use super::*;

  /// The x positions of the straight vertical strokes (`<line>`s and
  /// two-point `<polyline>`s) spanning at least 80% of the picture's
  /// height: the frame's sides, an axis and the vertical grid lines.
  fn full_height_verticals(svg: &str) -> Vec<f64> {
    let view_box = svg.split("viewBox=\"").nth(1).unwrap();
    let height: f64 = view_box[..view_box.find('"').unwrap()]
      .split(' ')
      .nth(3)
      .unwrap()
      .parse()
      .unwrap();
    let attr = |tag: &str, name: &str| -> Option<f64> {
      let start = tag.find(&format!(" {name}=\""))? + name.len() + 3;
      tag[start..start + tag[start..].find('"')?].parse().ok()
    };
    let mut xs = Vec::new();
    for tag in svg.split('<') {
      let segment = if tag.starts_with("line ") {
        (|| {
          Some((
            (attr(tag, "x1")?, attr(tag, "y1")?),
            (attr(tag, "x2")?, attr(tag, "y2")?),
          ))
        })()
      } else if tag.starts_with("polyline ") {
        let points: Vec<(f64, f64)> = tag
          .split(" points=\"")
          .nth(1)
          .and_then(|p| p.split('"').next())
          .unwrap_or("")
          .split_whitespace()
          .filter_map(|p| {
            let (x, y) = p.split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
          })
          .collect();
        (points.len() == 2).then(|| (points[0], points[1]))
      } else {
        None
      };
      if let Some(((x1, y1), (x2, y2))) = segment
        && (x1 - x2).abs() < 0.5
        && (y1 - y2).abs() > 0.8 * height
      {
        xs.push(x1);
      }
    }
    xs
  }

  // The y axis of a plot over `{x, 0, 10}` lies on the end of the unpadded
  // range, so the frame leaves it out; `Show` adding an option must not
  // bring it back.
  #[test]
  fn show_with_options_draws_the_plot_unchanged() {
    for plot in [
      "Plot[Sin[x], {x, 0, 10}, Frame -> True",
      "Plot[x^2 - 3, {x, 0, 4}, Frame -> True, PlotRange -> {-5, 5}",
      "Plot[Cos[x], {x, -3, 10}, Frame -> True",
    ] {
      assert_eq!(
        svg_of(&format!("Show[{plot}], GridLines -> Automatic]")),
        svg_of(&format!("{plot}, GridLines -> Automatic]")),
        "{plot}"
      );
    }
  }

  // Layering further plots keeps the leading plot's axes: none on the end
  // of its range, one where the origin is strictly inside.
  #[test]
  fn layered_plots_keep_the_leading_plots_axes() {
    let count = |code: &str| full_height_verticals(&svg_of(code)).len();
    // Nothing but what the plot draws on its own: no axis at x = 0.
    let plain = count("Plot[Sin[x], {x, 0, 10}, Frame -> True]");
    assert_eq!(
      count(
        "Show[Plot[Sin[x], {x, 0, 10}, Frame -> True], \
         Table[Plot[c, {x, 1, 12}], {c, {0.2, 0.5}}]]"
      ),
      plain
    );
    // The y axis at x = 0 as well.
    assert_eq!(
      count(
        "Show[Plot[Sin[x], {x, -3, 10}, Frame -> True], \
         Plot[0.5, {x, 1, 12}]]"
      ),
      plain + 1
    );
    // A `PlotRange` given to `Show` is the unpadded range itself.
    assert_eq!(
      count(
        "Show[Plot[Sin[x], {x, 0, 10}, Frame -> True], \
         Plot[0.5, {x, 1, 12}], PlotRange -> {{-2, 10}, {-2, 2}}]"
      ),
      plain + 1
    );
  }

  // The same holds when graphics primitives are layered over the plot,
  // which draws through the generic graphics renderer: no axis and no
  // automatic grid line on the ends of the plot's range.
  #[test]
  fn primitives_over_a_plot_keep_its_axes_and_grid_lines() {
    let count = |code: &str| full_height_verticals(&svg_of(code)).len();
    let disk = "Graphics[Disk[{5, 0}, 0.3]]";
    let framed = "Plot[Sin[x], {x, 0, 10}, Frame -> True]";
    let inner = "Plot[Sin[x], {x, -3, 10}, Frame -> True]";
    let plain = count(&format!("Show[{framed}, {disk}]"));
    assert_eq!(count(&format!("Show[{inner}, {disk}]")), plain + 1);
    // Grid lines at x = 2, 4, 6, 8 but not at 0 or 10.
    assert_eq!(
      count(&format!("Show[{framed}, {disk}, GridLines -> Automatic]")),
      plain + 4
    );
    // Explicit grid lines are drawn wherever they are asked for.
    assert_eq!(
      count(&format!(
        "Show[{framed}, {disk}, GridLines -> {{{{0, 5}}, None}}]"
      )),
      plain + 2
    );
    // Without a frame the axes are drawn as ever.
    let unframed = "Plot[Sin[x], {x, 0, 10}]";
    assert_eq!(
      count(&format!("Show[{unframed}, {disk}]")),
      count(&format!(
        "Show[{unframed}, {disk}, Axes -> {{True, False}}]"
      )) + 1
    );
  }
}

mod sized_tick_labels {
  use super::*;

  /// The baseline `y` of the `<text>` holding `content`, and the height of
  /// the picture in the same (viewBox) units.
  fn label_baseline(svg: &str, content: &str) -> (f64, f64) {
    let at = svg
      .find(content)
      .unwrap_or_else(|| panic!("{content} in {svg}"));
    let tag = &svg[svg[..at].rfind("<text ").unwrap()..at];
    let start = tag.find(" y=\"").unwrap() + 4;
    let y = tag[start..start + tag[start..].find('"').unwrap()]
      .parse()
      .unwrap();
    let view = svg.find("viewBox=\"").unwrap() + 9;
    let view = &svg[view..view + svg[view..].find('"').unwrap()];
    let height = view.split_whitespace().nth(3).unwrap().parse().unwrap();
    (y, height)
  }

  /// A tick label that sets its own size (`Style["a", 20]`) is drawn at that
  /// size in the plot's render units. Regression: the size was written in
  /// points into a picture ten units to the point, so the label was a tenth
  /// of its size and all but invisible.
  #[test]
  fn own_size_scales_with_the_rendering() {
    let ticks = "{{{1, Style[\"a\", 20]}}, {{1, Style[\"b\", 16]}}}";
    for plot in [
      format!("Plot[x^2, {{x, 0, 3}}, Ticks -> {ticks}]"),
      format!("ListPlot[{{1, 2, 3}}, Ticks -> {ticks}]"),
      format!("Plot[x^2, {{x, 0, 3}}, Frame -> True, FrameTicks -> {ticks}]"),
      format!(
        "ContourPlot[x y, {{x, 0, 3}}, {{y, 0, 3}}, FrameTicks -> {ticks}]"
      ),
    ] {
      let svg = svg_of(&plot);
      assert!(
        svg.contains("<tspan font-size=\"200\">a</tspan>"),
        "{plot}: {svg}"
      );
      assert!(
        svg.contains("<tspan font-size=\"160\">b</tspan>"),
        "{plot}: {svg}"
      );
      // The taller label still sits inside the picture.
      let (y, height) = label_baseline(&svg, "<tspan font-size=\"200\">a<");
      assert!(y < height, "{plot}: baseline {y} below {height}");
    }
  }

  /// The margin below the axis grows with the label drawn in it.
  #[test]
  fn larger_label_takes_more_room() {
    let height = |size: u32| {
      svg_size(&svg_of(&format!(
        "Plot[x^2, {{x, 0, 3}}, Ticks -> {{{{{{1, Style[\"a\", {size}]}}}}, \
         Automatic}}]"
      )))
      .1
    };
    assert!(height(40) > height(10));
  }
}

mod show_plot_range {
  use super::*;

  /// `Show` keeps the options of the plot it is given, its
  /// `PlotRangePadding` among them: a `PlotRange` given to `Show` is padded
  /// like the automatic range of that plot, where the same range given to
  /// the plot itself is drawn exactly.
  #[test]
  fn own_range_keeps_the_padding_of_the_plot() {
    let range = "PlotRange -> {{0, 1}, {0, 1}}";
    let shown = svg_of(&format!("Show[Plot[x, {{x, 0, 1}}], {range}]"));
    let exact = svg_of(&format!("Plot[x, {{x, 0, 1}}, {range}]"));
    let (shown_w, shown_h) = curve_extent(&shown);
    let (exact_w, exact_h) = curve_extent(&exact);
    // `Scaled[0.05]` above and below, and `Scaled[0.02]` on either side
    // (where the tick labels of the unpadded plot take room of their own).
    let ratio = shown_h / exact_h;
    assert!((0.85..0.97).contains(&ratio), "{shown_h} {exact_h}");
    assert!(shown_w < exact_w, "{shown_w} {exact_w}");
  }

  /// A plot drawn with an explicit range has no padding to pass on.
  #[test]
  fn plot_with_exact_range_stays_unpadded() {
    let plot = "Plot[x, {x, 0, 1}, PlotRange -> {{0, 1}, {0, 1}}]";
    let shown = svg_of(&format!(
      "Show[{plot}, PlotRange -> {{{{0, 1}}, {{0, 2}}}}]"
    ));
    let (shown_w, _) = curve_extent(&shown);
    let (exact_w, _) = curve_extent(&svg_of(plot));
    assert!(
      (shown_w / exact_w - 1.0).abs() < 0.005,
      "{shown_w} {exact_w}"
    );
  }
}

mod bar_accents {
  use super::*;

  /// `Overscript[X, Style["_", Bold, Red, 18]]` in a title of differently
  /// sized parts is a bar over the X in the accent's color. Regression: the
  /// styled accent was stacked above the base as an underscore glyph at its
  /// own size, a heavy dash floating over the letter.
  #[test]
  fn styled_bar_over_a_title_part_is_a_rule() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Row[{Style[\"a\", 12], \
       Style[Overscript[X, Style[\"_\", Bold, Red, 18]], Red, 14]}]]",
    );
    assert!(!svg.contains(">_</text>"), "{svg}");
    assert!(svg.contains("stroke=\"rgb(255,0,0)\""), "{svg}");
    // The bold accent draws a heavier rule than a plain one.
    let plain = svg_of(
      "Plot[x, {x, 0, 1}, PlotLabel -> Row[{Style[\"a\", 12], \
       Style[Overscript[X, \"_\"], Red, 14]}]]",
    );
    let rule_width = |svg: &str| -> f64 {
      let at = svg.find("fill=\"none\" stroke=").unwrap();
      let start = at + svg[at..].find("stroke-width=\"").unwrap() + 14;
      svg[start..start + svg[start..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    assert!(rule_width(&svg) > rule_width(&plain));
  }

  /// The base under a styled accent is set traditionally like any unstyled
  /// label: an italic X under its bar.
  #[test]
  fn base_under_a_styled_bar_is_italic() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, AxesLabel -> {Style[Overscript[X, \
       Style[\"_\", Bold, 18]], Red, 14], None}]",
    );
    assert!(
      svg.contains(
        "<tspan text-decoration=\"overline\"><tspan \
         font-style=\"italic\">X</tspan></tspan>"
      ),
      "{svg}"
    );
  }
}

mod epilog_text {
  use super::*;

  /// The styled parts of a `Text` row keep to its line: every nested
  /// `tspan` names the baseline itself, as resvg otherwise shifts it once
  /// more per level and lifts it above the plain parts beside it.
  #[test]
  fn nested_runs_name_their_baseline() {
    let svg = svg_of(
      "Plot[x, {x, 0, 1}, Epilog -> Text[Row[{Style[Subscript[Style[\"n\", \
       Italic], \"k\"], 12], \" = \", 1}], {0.5, 0.5}]]",
    );
    assert!(
      svg.contains(
        "<tspan dominant-baseline=\"middle\" font-style=\"italic\">n</tspan>"
      ),
      "{svg}"
    );
  }
}

mod explicit_axis_ticks {
  use super::*;

  /// The `<line>`s of a plot as `(x1, y1, x2, y2)`.
  fn lines(svg: &str) -> Vec<(f64, f64, f64, f64)> {
    svg
      .split("<line ")
      .skip(1)
      .filter_map(|tag| {
        let num = |key: &str| -> Option<f64> {
          tag
            .split(&format!("{key}=\""))
            .nth(1)?
            .split('"')
            .next()?
            .parse()
            .ok()
        };
        Some((num("x1")?, num("y1")?, num("x2")?, num("y2")?))
      })
      .collect()
  }

  /// The opening tag and content of the `<text>` holding `needle`.
  fn text_with<'a>(svg: &'a str, needle: &str) -> (&'a str, &'a str) {
    svg
      .split("<text ")
      .skip(1)
      .filter_map(|tag| tag.split_once('>'))
      .map(|(open, rest)| (open, rest.split("</text>").next().unwrap()))
      .find(|(_, content)| content.contains(needle))
      .unwrap_or_else(|| panic!("no text holding {needle:?}"))
  }

  /// `{pos, label, {0, 0}}` labels the axis without drawing a tick mark —
  /// the way an axis caption is written into a tick list. Regression: the
  /// length of a tick on an unframed axis was ignored, so the caption got a
  /// mark like every numbered tick.
  #[test]
  fn zero_length_tick_draws_its_label_without_a_mark() {
    let plot =
      |ticks: &str| svg_of(&format!("Plot[x, {{x, 0, 10}}, Ticks -> {ticks}]"));
    let marked = plot("{{2, {5, \"a\"}}, {4, {6, \"b\"}}}");
    for unmarked in [
      plot("{{2, {5, \"a\", {0, 0}}}, {4, {6, \"b\", {0, 0}}}}"),
      plot("{{2, {5, \"a\", 0}}, {4, {6, \"b\", 0}}}"),
    ] {
      assert_eq!(lines(&unmarked).len() + 2, lines(&marked).len());
      assert!(unmarked.contains(">a</text>") && unmarked.contains(">b</text>"));
    }
  }

  /// `{pos, label, {plen, mlen}}` reaches `plen` of the plot width into the
  /// plot and `mlen` out of it.
  #[test]
  fn tick_length_pair_reaches_into_and_out_of_the_plot() {
    let svg = svg_of(
      "Plot[x, {x, 0, 10}, Ticks -> {{2, {5, \"a\", {0.1, 0.05}}}, \
       {4, {6, \"b\", {0.1, 0.05}}}}]",
    );
    let all = lines(&svg);
    let longest = |vertical: bool| {
      all
        .iter()
        .copied()
        .filter(|l| (l.0 == l.2) == vertical)
        .max_by(|a, b| {
          let len = |l: &(f64, f64, f64, f64)| (l.2 - l.0).hypot(l.3 - l.1);
          len(a).total_cmp(&len(b))
        })
        .unwrap()
    };
    // The axes are the longest lines.
    let (x_axis, y_axis) = (longest(false), longest(true));
    let width = x_axis.2 - x_axis.0;
    let x_mark = all
      .iter()
      .find(|l| l.0 == l.2 && l.1 > x_axis.1 && l.3 < x_axis.1)
      .expect("an x tick mark crossing its axis");
    assert!((x_mark.1 - x_axis.1 - 0.05 * width).abs() < 1.0);
    assert!((x_axis.1 - x_mark.3 - 0.1 * width).abs() < 1.0);
    let y_mark = all
      .iter()
      .find(|l| **l != x_axis && l.1 == l.3 && l.0 < y_axis.0 && l.2 > y_axis.0)
      .expect("a y tick mark crossing its axis");
    assert!((y_axis.0 - y_mark.0 - 0.05 * width).abs() < 1.0);
    assert!((y_mark.2 - y_axis.0 - 0.1 * width).abs() < 1.0);
  }

  /// A tick label that starts with a line break is set one line below the
  /// other labels of its axis.
  #[test]
  fn tick_label_with_a_line_break_runs_over_two_lines() {
    let svg = svg_of(
      "Plot[x, {x, 0, 10}, \
       Ticks -> {{2, {5, Row[{\"\\n\", Style[\"V\", Italic]}]}}, Automatic}]",
    );
    let (_, content) = text_with(&svg, ">V<");
    let lines: Vec<&str> = content.split("<tspan x=").skip(1).collect();
    assert_eq!(lines.len(), 2, "{content}");
    assert!(lines[0].contains("dy=\"0.0\""));
    assert!(!lines[1].contains("dy=\"0.0\"") && lines[1].contains(">V<"));
  }

  /// Blanks padding a tick label keep their width, so they push a
  /// right-aligned label away from its axis.
  #[test]
  fn blanks_padding_a_tick_label_are_kept() {
    let svg = svg_of(
      "Plot[x, {x, 0, 10}, \
       Ticks -> {Automatic, {{4, \"plain\"}, {6, \"padded   \"}}}]",
    );
    assert!(
      text_with(&svg, "padded")
        .0
        .contains("xml:space=\"preserve\"")
    );
    assert!(!text_with(&svg, "plain").0.contains("xml:space"));
  }

  /// A plot label wrapped in `Text` is plain text: a letter in its box
  /// code stays upright, while the bare string is set in TraditionalForm
  /// with the letter in italics.
  #[test]
  fn text_plot_label_keeps_box_code_letters_upright() {
    let label = r#""of \!\(\*SubscriptBox[\(A\), \(1\)]\) used""#;
    let title = |wrapped: &str| {
      let svg =
        svg_of(&format!("Plot[x, {{x, 0, 1}}, PlotLabel -> {wrapped}]"));
      text_with(&svg, "used").1.to_string()
    };
    let sub = "<tspan baseline-shift=\"sub\" font-size=\"70%\">1</tspan>";
    assert_eq!(
      title(label),
      format!("of <tspan font-style=\"italic\">A</tspan>{sub} used")
    );
    let upright = format!("of A{sub} used");
    assert_eq!(title(&format!("Text[{label}]")), upright);
    assert_eq!(title(&format!("Style[Text[{label}], Red]")), upright);
  }
}

/// `Defer[expr]` in a label is typeset as `expr` alone, like `HoldForm`:
/// the wrapper only keeps the content unevaluated and is never drawn.
mod defer_labels {
  use super::svg_of;

  #[test]
  fn axes_labels_drop_the_defer_wrapper() {
    let svg =
      svg_of("Plot[t^2, {t, 0, 1}, AxesLabel -> {Defer[u], Defer[w^2]}]");
    assert!(!svg.contains("Defer"), "Defer wrapper was drawn");
    assert!(svg.contains(">u</text>"), "missing the x axis label");
    assert!(svg.contains(">w²</text>"), "missing the y axis label");
  }

  #[test]
  fn deferred_label_matches_the_held_one() {
    for body in ["u", "1 + 1", "a/b", "v = w + 2", "Sin[q]^2"] {
      let deferred = svg_of(&format!(
        "Plot[t, {{t, 0, 1}}, Frame -> True, PlotLabel -> Defer[{body}], \
         FrameLabel -> {{Defer[{body}], Defer[{body}]}}]"
      ));
      let held = svg_of(&format!(
        "Plot[t, {{t, 0, 1}}, Frame -> True, PlotLabel -> HoldForm[{body}], \
         FrameLabel -> {{HoldForm[{body}], HoldForm[{body}]}}]"
      ));
      assert_eq!(deferred, held, "label {body}");
    }
  }

  #[test]
  fn graphics_text_drops_the_defer_wrapper() {
    for body in ["u + 1", "2 + 2", "v = w + 2"] {
      let deferred =
        svg_of(&format!("Graphics[Text[Defer[{body}], {{0, 0}}]]"));
      let held = svg_of(&format!("Graphics[Text[HoldForm[{body}], {{0, 0}}]]"));
      assert!(!deferred.contains("Defer"), "Defer wrapper was drawn");
      assert_eq!(deferred, held, "text {body}");
    }
  }
}
