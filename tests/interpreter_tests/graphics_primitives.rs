use super::*;

mod graphics_primitives {
  use super::*;

  fn svg_of(expr: &str) -> String {
    let svg = interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap();
    assert!(svg.starts_with("<svg"), "no SVG for {expr}: {svg}");
    svg
  }

  /// The numbers of attribute `attr` of the first `<tag …>` element.
  fn attr_of(svg: &str, tag: &str, attr: &str) -> f64 {
    let start = svg.find(&format!("<{tag} ")).unwrap_or_else(|| {
      panic!("no <{tag}> in {svg}");
    });
    let el = &svg[start..start + svg[start..].find('>').unwrap()];
    let key = format!(" {attr}=\"");
    let i = el.find(&key).unwrap() + key.len();
    let j = el[i..].find('"').unwrap();
    el[i..i + j].parse().unwrap()
  }

  /// The points of the first `<polyline>`.
  fn polyline_points(svg: &str) -> Vec<(f64, f64)> {
    let i = svg.find("<polyline points=\"").unwrap() + 18;
    let j = svg[i..].find('"').unwrap();
    svg[i..i + j]
      .split_whitespace()
      .map(|p| {
        let (x, y) = p.split_once(',').unwrap();
        (x.parse().unwrap(), y.parse().unwrap())
      })
      .collect()
  }

  fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn annulus_draws_a_ring_with_a_hole() {
    let svg = svg_of("Graphics[{Blue, Annulus[{0, 0}, {0.5, 1}]}]");
    assert!(svg.contains("fill-rule=\"evenodd\""), "{svg}");
    assert!(svg.contains("fill=\"rgb(0,0,255)\""), "{svg}");
    // Two closed sub-paths: the outer circle and the hole.
    assert_eq!(svg.matches(" Z").count(), 2, "{svg}");
    // `Annulus[]` is the ring of radii 1/2 and 1 at the origin.
    let default = svg_of("Graphics[Annulus[]]");
    assert!(default.contains("fill-rule=\"evenodd\""), "{default}");
  }

  #[test]
  fn annulus_sector_draws_only_its_angles() {
    let svg = svg_of("Graphics[{Annulus[{0, 0}, {0.5, 1}, {0, Pi}]}]");
    assert!(svg.contains("<polygon"), "{svg}");
    // The upper half ring is twice as wide as it is tall.
    let w = attr_of(&svg, "svg", "width");
    let h = attr_of(&svg, "svg", "height");
    assert!(close(w / h, 2.0, 0.1), "{w} x {h}");
  }

  #[test]
  fn scaled_coordinates_resolve_against_the_plot_range() {
    let svg = svg_of(
      "Graphics[{Rectangle[{0, 0}, {10, 5}], Red, \
       Disk[Scaled[{0.5, 0.5}], Scaled[0.05]], \
       Line[{Scaled[{0, 0}], Scaled[{1, 1}]}]}]",
    );
    assert!(!svg.contains("should be a pair of numbers"), "{svg}");
    let w = attr_of(&svg, "svg", "width");
    // The disk sits in the middle, its radii 5% of each side.
    let cx = attr_of(&svg, "ellipse", "cx");
    let rx = attr_of(&svg, "ellipse", "rx");
    let ry = attr_of(&svg, "ellipse", "ry");
    assert!(close(cx, w / 2.0, 1.5), "{svg}");
    // Radii are 5% of the drawing area's width and height (359 x 186.11).
    assert!(close(rx, 0.05 * 359.0, 0.05), "{svg}");
    assert!(close(ry, 0.05 * 186.11, 0.05), "{svg}");
    // The line runs corner to corner.
    let pts = polyline_points(&svg);
    assert!(
      close(pts[0].0, 0.0, 0.5) && close(pts[1].1, 0.0, 0.5),
      "{pts:?}"
    );
    assert!(close(pts[1].0, w - 1.0, 1.0), "{pts:?}");
  }

  #[test]
  fn scaled_rectangle_corners() {
    let svg = svg_of(
      "Graphics[{Rectangle[{0, 0}, {10, 5}], \
       Green, Rectangle[Scaled[{0.8, 0.1}], Scaled[{0.95, 0.3}]]}]",
    );
    let green = svg.find("fill=\"rgb(0,255,0)\"").unwrap();
    let x = attr_of(&svg[svg[..green].rfind("<rect").unwrap()..], "rect", "x");
    let w = attr_of(&svg, "svg", "width");
    assert!(close(x / w, 0.8, 0.01), "{svg}");
  }

  #[test]
  fn offset_coordinates_move_by_printer_points() {
    let svg = svg_of(
      "Graphics[{Point[{1, 1}], Line[{{1, 1}, Offset[{30, 0}, {1, 1}]}]}, \
       PlotRange -> {{0, 3}, {0, 2}}]",
    );
    let pts = polyline_points(&svg);
    assert!(close(pts[1].0 - pts[0].0, 30.0, 0.05), "{pts:?}");
    assert!(close(pts[1].1, pts[0].1, 0.01), "{pts:?}");
  }

  #[test]
  fn image_scaled_coordinates_resolve_against_the_whole_image() {
    let svg = svg_of(
      "Graphics[{Disk[], Red, Disk[ImageScaled[{0.1, 0.9}], 0.1]}, \
       Frame -> True]",
    );
    let w = attr_of(&svg, "svg", "width");
    let h = attr_of(&svg, "svg", "height");
    let t = svg.find("translate(").unwrap() + 10;
    let (dx, rest) = svg[t..].split_once(',').unwrap();
    let dy: f64 = rest[..rest.find(')').unwrap()].parse().unwrap();
    let dx: f64 = dx.parse().unwrap();
    let red = &svg[svg.find("fill=\"rgb(255,0,0)\"").unwrap() - 80..];
    let cx = attr_of(red, "ellipse", "cx");
    let cy = attr_of(red, "ellipse", "cy");
    assert!(close(cx + dx, 0.1 * w, 0.5), "{svg}");
    assert!(close(cy + dy, 0.1 * h, 0.5), "{svg}");
  }

  #[test]
  fn bezier_curve_spline_degree_one_is_its_control_polygon() {
    let svg = svg_of(
      "Graphics[BezierCurve[{{0, 0}, {1, 2}, {2, 0}}, SplineDegree -> 1]]",
    );
    assert_eq!(polyline_points(&svg).len(), 3, "{svg}");
  }

  #[test]
  fn bezier_curve_spline_degree_changes_the_curve() {
    let pts = "{{0, 0}, {1, 2}, {2, 0}, {3, 2}, {4, 0}}";
    let quadratic =
      svg_of(&format!("Graphics[BezierCurve[{pts}, SplineDegree -> 2]]"));
    let quartic =
      svg_of(&format!("Graphics[BezierCurve[{pts}, SplineDegree -> 4]]"));
    // The composite quadratic touches the control point {2, 0}; the
    // single quartic does not.
    let touches = |svg: &str| {
      let p = polyline_points(svg);
      let bottom = p.iter().map(|q| q.1).fold(f64::MIN, f64::max);
      p.iter()
        .filter(|q| close(q.1, bottom, 0.01))
        .any(|q| q.0 > 100.0 && q.0 < 260.0)
    };
    assert!(touches(&quadratic), "{quadratic}");
    assert!(!touches(&quartic), "{quartic}");
  }

  #[test]
  fn bspline_curve_spline_weights_pull_the_curve() {
    let pts = "{{0, 0}, {1, 2}, {2, 0}, {3, 2}, {4, 0}}";
    let mid_y = |svg: &str| {
      let p = polyline_points(svg);
      p[p.len() / 2].1
    };
    let plain =
      svg_of(&format!("Graphics[{{Point[{pts}], BSplineCurve[{pts}]}}]"));
    let weighted = svg_of(&format!(
      "Graphics[{{Point[{pts}], BSplineCurve[{pts}, \
       SplineWeights -> {{1, 1, 5, 1, 1}}]}}]"
    ));
    // A heavy middle point (at the bottom) pulls the curve down.
    assert!(mid_y(&weighted) > mid_y(&plain) + 20.0);
  }

  #[test]
  fn bspline_curve_degree_above_three() {
    let pts = "{{0, 0}, {1, 2}, {2, -1}, {3, 2}, {4, 0}, {5, 1}}";
    let cubic =
      svg_of(&format!("Graphics[{{Point[{pts}], BSplineCurve[{pts}]}}]"));
    let quintic = svg_of(&format!(
      "Graphics[{{Point[{pts}], BSplineCurve[{pts}, SplineDegree -> 5]}}]"
    ));
    assert_ne!(polyline_points(&cubic), polyline_points(&quintic));
  }

  #[test]
  fn joined_curve_continues_each_piece_from_the_previous_end() {
    let svg = svg_of(
      "Graphics[JoinedCurve[{Line[{{0, 0}, {2, 0}}], \
       BezierCurve[{{2, 0}, {3, 0}, {3, 2}, {2, 2}}], \
       Line[{{2, 2}, {0, 2}}]}, CurveClosed -> True]]",
    );
    let pts = polyline_points(&svg);
    // Closed back to the start.
    assert_eq!(pts.first(), pts.last());
    // The Bezier piece starts at {2, 0} and its control points are all
    // four given ones, so it reaches x = 3 (the right edge).
    let right = pts.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    let w = attr_of(&svg, "svg", "width");
    assert!(right > w - 12.0, "{svg}");
  }

  #[test]
  fn filled_curve_joins_its_pieces_into_one_boundary() {
    let svg = svg_of(
      "Graphics[{Orange, FilledCurve[{Line[{{0, 0}, {2, 0}}], \
       BezierCurve[{{2, 0}, {3, 1}, {2, 2}}], \
       BSplineCurve[{{2, 2}, {1, 3}, {0, 2}}], Line[{{0, 2}, {0, 0}}]}]}]",
    );
    assert_eq!(svg.matches("<polygon").count(), 1, "{svg}");
  }

  #[test]
  fn lch_and_luv_colors() {
    // Out-of-gamut channels clip.
    assert!(
      interpret("ColorConvert[LCHColor[0.6, 0.5, 0.7], \"RGB\"]")
        .unwrap()
        .starts_with("RGBColor[0., 0.608")
    );
    let luv = interpret(
      "Round[255 List @@ ColorConvert[LUVColor[0.7, 0.3, -0.5], \"RGB\"]]",
    )
    .unwrap();
    assert_eq!(luv, "{210, 149, 232}");
    let svg = svg_of("Graphics[{LCHColor[0.6, 0.5, 0.7], Rectangle[]}]");
    assert!(svg.contains("fill=\"rgb(0,155,228)\""), "{svg}");
  }

  #[test]
  fn raster_color_function() {
    let svg = svg_of(
      "Graphics[Raster[{{0, 1}}, ColorFunction -> (Blend[{Blue, Yellow}, #] &)]]",
    );
    assert!(svg.contains("rgb(0,0,255)"), "{svg}");
    assert!(svg.contains("rgb(255,255,0)"), "{svg}");
    // With an explicit value range a function is handed values rescaled
    // from the data's own extremes; a color head gets the range itself.
    let blend = svg_of(
      "Graphics[Raster[{{0, 10, 20, 30}}, {{0, 0}, {4, 1}}, {0, 40}, \
       ColorFunction -> (GrayLevel[#] &)]]",
    );
    assert!(blend.contains("rgb(255,255,255)"), "{blend}");
    let head = svg_of(
      "Graphics[Raster[{{0, 10, 20, 30}}, {{0, 0}, {4, 1}}, {0, 40}, \
       ColorFunction -> GrayLevel]]",
    );
    assert!(head.contains("rgb(191,191,191)"), "{head}");
    assert!(!head.contains("rgb(255,255,255)"), "{head}");
  }

  // A color scheme given by name colors the cells like the `ColorData`
  // gradient of that name, clipping values beyond the scheme's range.
  #[test]
  fn raster_color_function_named_scheme() {
    for (data, range) in [
      ("{{0, 0.5}, {1, 10000}}", ""),
      ("{{0, 5}, {10, 20}}", ", {{0, 0}, {2, 2}}, {0, 10}"),
    ] {
      for scheme in ["ThermometerColors", "Rainbow", "SunsetColors"] {
        let named = svg_of(&format!(
          "Graphics[Raster[{data}{range}, ColorFunction -> \"{scheme}\"]]"
        ));
        let explicit = svg_of(&format!(
          "Graphics[Raster[{data}{range}, \
           ColorFunction -> ColorData[\"{scheme}\"]]]"
        ));
        assert_eq!(named, explicit, "{scheme}");
      }
    }
    let svg = svg_of(
      "Graphics[Raster[{{0, 0.5}, {1, 10000}}, \
       ColorFunction -> \"ThermometerColors\"]]",
    );
    assert!(svg.contains("rgb(42,31,202)"), "{svg}");
    assert!(svg.contains("rgb(220,227,209)"), "{svg}");
    assert_eq!(svg.matches("rgb(136,22,43)").count(), 2, "{svg}");
  }

  // Scaling from the default `{0, 1}` range is the identity, so a color
  // function sees the values as given: a stored indexed `RasterBox` looks
  // its integer cells up in a palette, and a plain number from the color
  // function is a gray level.
  #[test]
  fn raster_color_function_sees_unscaled_indices() {
    let svg = svg_of(
      "Graphics[Raster[{{1, 2}}, {{0, 0}, {2, 1}}, \
       ColorFunction -> (Apply[GrayLevel, Part[{0., 65535.}, #]] &)]]",
    );
    assert!(svg.contains("fill=\"rgb(0,0,0)\""), "{svg}");
    assert!(svg.contains("fill=\"rgb(255,255,255)\""), "{svg}");
    let palette = svg_of(
      "Graphics[Raster[{{1, 2, 3}}, {{0, 0}, {3, 1}}, \
       ColorFunction -> ({Red, Green, Blue}[[#]] &)]]",
    );
    assert!(palette.contains("fill=\"rgb(255,0,0)\""), "{palette}");
    assert!(palette.contains("fill=\"rgb(0,255,0)\""), "{palette}");
    assert!(palette.contains("fill=\"rgb(0,0,255)\""), "{palette}");
  }

  #[test]
  fn frame_style_per_edge() {
    let svg = svg_of(
      "Graphics[{Gray, Disk[]}, Frame -> True, FrameStyle -> \
       {{Directive[Red, Thick], Blue}, {Directive[Darker[Green], Dashed], Orange}}]",
    );
    // Each edge is drawn on its own in its style.
    assert!(
      !svg
        .split("<rect ")
        .any(|r| r.starts_with("x=\"0\"") && r.contains("fill=\"none\"")),
      "{svg}"
    );
    let edge = |color: &str| {
      svg.split("<line ").any(|l| {
        l.contains(&format!("stroke=\"{color}\""))
          && l.contains("stroke-width=\"2\"")
      })
    };
    assert!(edge("rgb(255,0,0)"), "thick red left edge: {svg}");
    assert!(svg.contains("stroke=\"rgb(0,0,255)\""), "{svg}");
    assert!(svg.contains("stroke=\"rgb(255,128,0)\""), "{svg}");
    assert!(
      svg.contains("stroke-dasharray"),
      "dashed bottom edge: {svg}"
    );
    // Tick labels take their edge's color.
    assert!(svg.contains("fill=\"rgb(255,0,0)\""), "{svg}");
    assert!(svg.contains("fill=\"rgb(0,170,0)\""), "{svg}");
  }

  #[test]
  fn frame_ticks_outside_the_plot_range_are_dropped() {
    let svg = svg_of(
      "Graphics[{Red, Disk[{1, 1}, 0.8]}, Frame -> True, FrameTicks -> \
       {{{0, 1, 2}, None}, {{{0, \"zero\"}, {1, \"one\"}, {2, \"two\"}}, None}}]",
    );
    assert!(svg.contains(">one<"), "{svg}");
    assert!(!svg.contains(">zero<") && !svg.contains(">two<"), "{svg}");
    assert!(!svg.contains(">0<") && !svg.contains(">2<"), "{svg}");
  }

  #[test]
  fn automatic_frame_ticks_have_minor_marks() {
    let svg = svg_of("Graphics[Rectangle[{0, 0}, {2, 1}], Frame -> True]");
    let lens: Vec<f64> = svg
      .split("<line x1=\"")
      .skip(1)
      .filter_map(|l| {
        let num = |key: &str| -> Option<f64> {
          let i = l.find(key)? + key.len();
          l[i..i + l[i..].find('"')?].parse().ok()
        };
        let (y1, y2) = (num(" y1=\"")?, num(" y2=\"")?);
        Some((y1 - y2).abs())
      })
      .filter(|d| *d > 0.0)
      .collect();
    // Labeled marks 1.5% of the geometric mean of the frame's sides,
    // unlabeled ones three fifths of that.
    let major = lens.iter().copied().fold(0.0, f64::max);
    assert!(
      lens.iter().any(|d| (d - 0.6 * major).abs() < 0.02),
      "{lens:?}"
    );
    assert!(major > 2.0 && major < 4.0, "{major}");
  }

  #[test]
  fn face_form_list_with_opacity() {
    let svg =
      svg_of("Graphics[{FaceForm[{Opacity[0.3], Green}], Rectangle[]}]");
    assert!(svg.contains("fill=\"rgb(0,255,0)\""), "{svg}");
    assert!(svg.contains("fill-opacity=\"0.3\""), "{svg}");
  }

  #[test]
  fn grid_lines_inherit_grid_lines_style() {
    let svg = svg_of(
      "Graphics[Point[{1, 1}], PlotRange -> {{0, 3}, {0, 2}}, \
       GridLines -> {{1, {2, Directive[Red, Thick]}}, {0.5}}, \
       GridLinesStyle -> Directive[Blue, Dashed]]",
    );
    let lines: Vec<&str> = svg.split("<line ").skip(1).collect();
    let red = lines.iter().find(|l| l.contains("rgb(255,0,0)")).unwrap();
    assert!(red.contains("stroke-dasharray"), "{red}");
    let blue = lines.iter().find(|l| l.contains("rgb(0,0,255)")).unwrap();
    assert!(blue.contains("stroke-width=\"0.20\""), "{blue}");
    // Without a style: thin mid gray.
    let plain = svg_of(
      "Graphics[Point[{1, 1}], PlotRange -> {{0, 3}, {0, 2}}, GridLines -> {{1}, {}}]",
    );
    assert!(
      plain.contains("stroke=\"rgb(128,128,128)\" stroke-width=\"0.20\""),
      "{plain}"
    );
  }

  #[test]
  fn self_intersecting_polygon_fills_even_odd() {
    let star = svg_of(
      "Graphics[Polygon[Table[{Cos[2 Pi k 2/5 + Pi/2], Sin[2 Pi k 2/5 + Pi/2]}, {k, 0, 4}]]]",
    );
    assert!(star.contains("fill-rule=\"evenodd\""), "{star}");
    let square = svg_of("Graphics[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]");
    assert!(!square.contains("fill-rule"), "{square}");
  }

  /// Twice around from the same corner: a square, then a smaller loop
  /// inside it that starts where the square did.
  const NESTED_LOOPS: &str = "{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {0, 0}, \
     {3, 1}, {3, 3}, {1, 3}}";

  #[test]
  fn polygon_revisiting_a_corner_fills_even_odd() {
    let svg = svg_of(&format!("Graphics[Polygon[{NESTED_LOOPS}]]"));
    assert!(svg.contains("fill-rule=\"evenodd\""), "{svg}");
    // A doubled corner or a closing repeat of the first one is no loop.
    let closed =
      svg_of("Graphics[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}]]");
    assert!(!closed.contains("fill-rule"), "{closed}");
    let doubled =
      svg_of("Graphics[Polygon[{{0, 0}, {1, 0}, {1, 0}, {1, 1}, {0, 1}}]]");
    assert!(!doubled.contains("fill-rule"), "{doubled}");
  }

  /// The centroids of the `<polygon>` facets of a vertex-colored fill, as
  /// fractions of the facets' joint bounding box (y measured upward).
  fn facet_centroids(svg: &str) -> Vec<(f64, f64)> {
    let tris: Vec<Vec<(f64, f64)>> = svg
      .split("<polygon points=\"")
      .skip(1)
      .map(|rest| {
        rest[..rest.find('"').unwrap()]
          .split_whitespace()
          .map(|p| {
            let (x, y) = p.split_once(',').unwrap();
            (x.parse().unwrap(), y.parse().unwrap())
          })
          .collect()
      })
      .collect();
    assert!(!tris.is_empty(), "{svg}");
    let all = tris.iter().flatten();
    let (x0, x1, y0, y1) = all.fold(
      (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
      |(a, b, c, d), &(x, y)| (a.min(x), b.max(x), c.min(y), d.max(y)),
    );
    tris
      .iter()
      .map(|t| {
        let k = t.len() as f64;
        let cx = t.iter().map(|p| p.0).sum::<f64>() / k;
        let cy = t.iter().map(|p| p.1).sum::<f64>() / k;
        ((cx - x0) / (x1 - x0), (y1 - cy) / (y1 - y0))
      })
      .collect()
  }

  #[test]
  fn vertex_colored_polygon_leaves_a_doubly_wound_loop_empty() {
    // The inner loop is wound twice, so the even-odd rule leaves it empty
    // even though every point carries a color.
    let svg = svg_of(&format!(
      "Graphics[Polygon[{NESTED_LOOPS}, VertexColors -> Table[Hue[k/8], {{k, 8}}]]]"
    ));
    let inside =
      |(x, y): (f64, f64)| (0.45..0.7).contains(&x) && (0.4..0.65).contains(&y);
    let centroids = facet_centroids(&svg);
    assert!(!centroids.iter().any(|&c| inside(c)), "{svg}");
    // The once-wound band between the loops is still filled.
    assert!(centroids.iter().any(|&(x, y)| x > 0.8 && y > 0.3), "{svg}");
  }

  #[test]
  fn vertex_colored_pentagram_leaves_its_center_empty() {
    let svg = svg_of(
      "Graphics[Polygon[Table[{Cos[4 Pi k/5 + Pi/2], Sin[4 Pi k/5 + Pi/2]}, \
       {k, 0, 4}], VertexColors -> {Red, Green, Blue, Yellow, Cyan}]]",
    );
    let centroids = facet_centroids(&svg);
    // The pentagon in the middle (around the center of the star, which
    // sits at 45% of its height) is wound twice.
    let center =
      |(x, y): (f64, f64)| (0.4..0.6).contains(&x) && (0.37..0.53).contains(&y);
    assert!(!centroids.iter().any(|&c| center(c)), "{svg}");
    // The tips are filled.
    assert!(centroids.iter().any(|&(_, y)| y > 0.9), "{svg}");
  }

  #[test]
  fn points_at_the_range_edge_stay_whole() {
    let svg =
      svg_of("Graphics[{PointSize[0.05], Point[{{0, 0}, {1, 1}, {2, 0}}]}]");
    let cx = attr_of(&svg, "circle", "cx");
    let r = attr_of(&svg, "circle", "r");
    assert!(cx >= r - 0.01, "{svg}");
  }

  #[test]
  fn cap_form_applies_to_circle_arcs() {
    let svg = svg_of(
      "Graphics[{CapForm[\"Round\"], Thickness[0.02], Circle[{0, 0}, 1, {Pi, 2 Pi}]}]",
    );
    assert!(svg.contains("stroke-linecap=\"round\""), "{svg}");
  }

  #[test]
  fn reversed_circle_arc_range_draws_the_same_arc() {
    // `{a1, a2}` with `a1 > a2` covers the same angles as `{a2, a1}`.
    for (rev, fwd) in [("{3, 1}", "{1, 3}"), ("{5.8, 2.7}", "{2.7, 5.8}")] {
      let path = |range: &str| {
        let svg = svg_of(&format!(
          "Graphics[Circle[{{0, 0}}, 1, {range}], PlotRange -> 1.2]"
        ));
        let start = svg.find("<path").expect("arc path");
        svg[start..].lines().next().unwrap().to_string()
      };
      assert_eq!(path(rev), path(fwd));
    }
  }

  #[test]
  fn reversed_disk_sector_range_draws_the_same_sector() {
    for (rev, fwd) in [("{3, 1}", "{1, 3}"), ("{5.8, 2.7}", "{2.7, 5.8}")] {
      let path = |range: &str| {
        let svg = svg_of(&format!(
          "Graphics[Disk[{{0, 0}}, 1, {range}], PlotRange -> 1.2]"
        ));
        let start = svg.find("<path").expect("sector path");
        svg[start..].lines().next().unwrap().to_string()
      };
      assert_eq!(path(rev), path(fwd));
    }
  }

  #[test]
  fn arrowhead_graphics_are_scaled_by_the_head_size() {
    let svg = svg_of(
      "Graphics[{Arrowheads[{{0.1, 1, Graphics[Disk[{0, 0}, 0.5]]}}], \
       Arrow[{{0, 0}, {3, 0}}]}, PlotRange -> {{-1, 4}, {-1, 1}}]",
    );
    // One unit of the head graphic is 10% of the drawing width (359).
    let rx = attr_of(&svg, "ellipse", "rx");
    assert!(close(rx, 0.05 * 359.0, 0.1), "{svg}");
  }

  #[test]
  fn delaunay_and_voronoi_meshes_take_mesh_cell_style() {
    let d = interpret(
      "Head[DelaunayMesh[{{0., 0.}, {1., 0.}, {0., 1.}, {1., 1.2}}, \
       MeshCellStyle -> {{2, All} -> Red}]]",
    )
    .unwrap();
    assert_eq!(d, "MeshRegion");
    let svg = svg_of(
      "DelaunayMesh[{{0., 0.}, {1., 0.}, {0., 1.}, {1., 1.2}}, \
       MeshCellStyle -> {{2, All} -> Red, {1, All} -> Black, {0, All} -> Blue}]",
    );
    assert!(svg.contains("fill=\"rgb(255,0,0)\""), "{svg}");
    assert!(svg.contains("stroke=\"rgb(0,0,0)\""), "{svg}");
    assert!(svg.contains("fill=\"rgb(0,0,255)\""), "points: {svg}");
    let v = svg_of(
      "VoronoiMesh[{{0.1, 0.2}, {0.4, 0.8}, {0.7, 0.3}, {0.5, 0.5}}, \
       {{0, 1}, {0, 1}}, MeshCellStyle -> {2 -> Red}]",
    );
    assert!(v.contains("fill=\"rgb(255,0,0)\""), "{v}");
  }

  #[test]
  fn voronoi_cells_stay_inside_their_bounds() {
    // The cell of the interior site {0.4, 0.8} has a corner past y = 1.
    let corners = interpret(
      "pts = {{0.1, 0.2}, {0.4, 0.8}, {0.7, 0.3}, {0.9, 0.9}, {0.5, 0.5}, \
       {0.2, 0.6}, {0.8, 0.6}, {0.3, 0.1}, {0.6, 0.05}, {0.15, 0.95}}; \
       MinMax[MeshCoordinates[VoronoiMesh[pts, {{0, 1}, {0, 1}}]]]",
    )
    .unwrap();
    assert_eq!(corners, "{0., 1.}");
  }

  #[test]
  fn scaled_plot_range_padding_is_a_share_of_the_padded_range() {
    // Scaled[0.2] on both sides of the span 2 pads it to 2 / 0.6: the
    // unit disk spans 60% of the drawing width.
    let svg = svg_of("Graphics[Disk[], PlotRangePadding -> Scaled[0.2]]");
    let rx = attr_of(&svg, "ellipse", "rx");
    let w = attr_of(&svg, "svg", "width");
    assert!(close(2.0 * rx / (w - 1.0), 0.6, 0.01), "{svg}");
  }

  #[test]
  fn automatic_plot_range_padding_leaves_explicit_ranges_unpadded() {
    // `Automatic` pads only ranges determined from the primitives, like
    // the default: a range pinned by `PlotRange` is drawn exactly.
    for range in ["{{-1, 1}, {-1, 1}}", "{{-3, 2}, {-1, 4}}"] {
      let plain = svg_of(&format!(
        "Graphics[Disk[], PlotRange -> {range}, Axes -> True]"
      ));
      for padding in ["Automatic", "{Automatic, Automatic}"] {
        let padded = svg_of(&format!(
          "Graphics[Disk[], PlotRange -> {range}, Axes -> True, \
           PlotRangePadding -> {padding}]"
        ));
        assert_eq!(padded, plain, "PlotRangePadding -> {padding}");
      }
    }
    // Per axis: the Scaled x side pads, the Automatic y side does not, so
    // the unit disk fills the height but only 60% of the padded width.
    let svg = svg_of(
      "Graphics[Disk[], PlotRange -> {{-1, 1}, {-1, 1}}, \
       PlotRangePadding -> {Scaled[0.2], Automatic}, AspectRatio -> Full, \
       ImageSize -> {300, 300}]",
    );
    let rx = attr_of(&svg, "ellipse", "rx");
    let ry = attr_of(&svg, "ellipse", "ry");
    assert!(close(ry / rx, 1.0 / 0.6, 0.02), "{svg}");
  }

  #[test]
  fn primitives_spill_past_a_frame_into_the_margins() {
    // PlotRangeClipping -> False is the default: only the image cuts.
    let svg = svg_of(
      "Graphics[{Red, Disk[{0, 0}, 2]}, PlotRange -> {{-1, 2}, {-1, 2}}, \
       Frame -> True]",
    );
    assert!(!svg.contains("<g clip-path"), "{svg}");
    let clipped = svg_of(
      "Graphics[{Red, Disk[{0, 0}, 2]}, PlotRange -> {{-1, 2}, {-1, 2}}, \
       Frame -> True, PlotRangeClipping -> True]",
    );
    assert!(clipped.contains("<g clip-path"), "{clipped}");
  }

  #[test]
  fn axes_labels_take_the_axes_style_color() {
    let svg = svg_of(
      "Graphics[Circle[{1, 1}, 1], Axes -> True, AxesLabel -> {\"x\", \"y\"}, \
       AxesStyle -> {Blue, Darker[Green]}]",
    );
    assert!(
      svg.contains(">x</text>") && svg.contains(">y</text>"),
      "{svg}"
    );
    let x = &svg[..svg.find(">x</text>").unwrap()];
    assert!(
      x[x.rfind("<text").unwrap()..].contains("rgb(0,0,255)"),
      "{svg}"
    );
    let y = &svg[..svg.find(">y</text>").unwrap()];
    assert!(
      y[y.rfind("<text").unwrap()..].contains("rgb(0,170,0)"),
      "{svg}"
    );
  }

  #[test]
  fn named_arrowhead_sizes_are_absolute() {
    // Large is 20 points long at any image size.
    for width in [200, 400] {
      let svg = svg_of(&format!(
        "Graphics[{{Arrowheads[Large], Arrow[{{{{0, 0}}, {{3, 0}}}}]}}, \
         ImageSize -> {width}]"
      ));
      let i = svg.find("<polygon points=\"").unwrap() + 17;
      let pts: Vec<f64> = svg[i..i + svg[i..].find('"').unwrap()]
        .split([' ', ','])
        .map(|v| v.parse().unwrap())
        .collect();
      assert!(close(pts[0] - pts[2], 20.0, 0.05), "{svg}");
    }
  }

  #[test]
  fn arrowheads_widen_the_plot_range() {
    // A head at the start of the arrow points past it; it is not cut off.
    let svg =
      svg_of("Graphics[{Arrowheads[{{0.05, 0}}], Arrow[{{0, 0}, {3, 0}}]}]");
    let i = svg.find("<polygon points=\"").unwrap() + 17;
    let xs: Vec<f64> = svg[i..i + svg[i..].find('"').unwrap()]
      .split(' ')
      .map(|p| p.split(',').next().unwrap().parse().unwrap())
      .collect();
    assert!(xs.iter().all(|&x| x >= -0.01), "{svg}");
  }

  #[test]
  fn axes_skip_the_label_where_they_cross() {
    let svg = svg_of(
      "Graphics[Line[{{0, 0}, {4, 3}}], Axes -> True, AxesOrigin -> {1, 1}]",
    );
    // 0 is labeled on the x axis, 1 (the crossing) on neither.
    assert!(svg.contains(">0</text>"), "{svg}");
    assert!(
      svg.contains(">0.0</text>") || svg.contains(">0.</text>"),
      "{svg}"
    );
    assert!(
      !svg.contains(">1</text>") && !svg.contains(">1.0</text>"),
      "{svg}"
    );
  }

  #[test]
  fn named_dashing_sizes() {
    let svg = svg_of(
      "Graphics[{Dashing[Medium], Line[{{0, 0}, {1, 0}}], \
       Dashing[Large], Line[{{0, 1}, {1, 1}}]}]",
    );
    assert!(svg.contains("stroke-dasharray=\"5.0,7.0\""), "{svg}");
    assert!(svg.contains("stroke-dasharray=\"10.0,13.0\""), "{svg}");
  }

  #[test]
  fn text_background_hugs_the_text() {
    let svg = svg_of(
      "Graphics[{LightGray, Rectangle[{0, 0}, {4, 3}], \
       Text[Style[\"Style background\", Background -> Yellow], {2, 2}]}]",
    );
    let yellow = svg.find("fill=\"rgb(255,255,0)\"").unwrap();
    let rect = &svg[svg[..yellow].rfind("<rect").unwrap()..];
    let w = attr_of(rect, "rect", "width");
    // 16 characters of 10-point type: about 77 points of advance.
    assert!(w > 68.0 && w < 90.0, "{w}");
  }

  #[test]
  fn vertex_color_gradient_has_no_seams() {
    let svg = svg_of(
      "Graphics[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, \
       VertexColors -> {Red, Green, Blue, Yellow}]]",
    );
    // Opaque facets are outlined in their own color.
    let facets = svg.matches("<polygon").count();
    assert!(facets > 100, "{facets}");
    let outlined = svg.matches("stroke-width=\"0.5\"").count();
    assert!(
      outlined + 1 >= facets && outlined > 100,
      "{outlined} of {facets}"
    );
  }
  /// A label anchored outside an explicit `PlotRange` lies off the canvas,
  /// as in Wolfram; it is not pinned to the image edge. Regression: the
  /// pass that keeps tick labels inside the canvas dragged it to `x = 4.2`.
  #[test]
  fn text_outside_explicit_plot_range_stays_out_of_view() {
    let svg = svg_of(
      "Graphics[{Circle[], Text[\"A\", {-7, 0.5}], Text[\"B\", {7, 0.5}, \
       {0, -2}]}, PlotRange -> {{-2, 2}, {-2, 2}}, ImageSize -> 200]",
    );
    let a = &svg[svg
      .find(">A<")
      .map(|e| svg[..e].rfind("<text").unwrap())
      .unwrap()..];
    let b = &svg[svg
      .find(">B<")
      .map(|e| svg[..e].rfind("<text").unwrap())
      .unwrap()..];
    let ax = attr_of(a, "text", "x");
    let bx = attr_of(b, "text", "x");
    assert!(ax < -200.0, "{ax}");
    assert!(bx > 400.0, "{bx}");
  }

  /// Every `<polyline>`'s `stroke-linecap`.
  fn polyline_caps(svg: &str) -> Vec<String> {
    svg
      .match_indices("<polyline ")
      .map(|(i, _)| {
        let el = &svg[i..i + svg[i..].find('>').unwrap()];
        let k = el.find("stroke-linecap=\"").unwrap() + 16;
        el[k..k + el[k..].find('"').unwrap()].to_string()
      })
      .collect()
  }

  /// A dashed stroke is drawn with butt caps whatever its `CapForm`, as in
  /// Wolfram's SVG export: `Dashing[{0, d}]` with round caps gave fat
  /// round dots, where Wolfram draws one-pixel ticks; a dashed `Line`
  /// without a `CapForm` lost its square caps' lengthening of every dash.
  #[test]
  fn dashed_lines_have_butt_caps() {
    let svg = svg_of(
      "Graphics[{{Dashing[{0, 0.02}], CapForm[\"Round\"], \
       AbsoluteThickness[4], Line[{{0, 0}, {4, 0}}]}, \
       {Dashed, Line[{{0, 1}, {4, 1}}]}, Line[{{0, 2}, {4, 2}}]}]",
    );
    assert_eq!(polyline_caps(&svg), ["butt", "butt", "square"], "{svg}");
    assert!(svg.contains("stroke-dasharray=\"1.0,"), "{svg}");
  }

  /// A thick line keeps its stroke inside the picture: Wolfram widens the
  /// margin to half the stroke width past each vertex (15 points for a
  /// 30-point line, even with `PlotRangePadding -> None`), so round caps
  /// are not cut off at the image edge.
  #[test]
  fn thick_line_margin_holds_half_its_stroke() {
    for cap in ["Butt", "Round"] {
      let svg = svg_of(&format!(
        "Graphics[{{AbsoluteThickness[30], CapForm[\"{cap}\"], \
         Line[{{{{0, 0}}, {{3, 0}}}}], Thin, Line[{{{{0, -1}}, {{0, 2}}}}]}}, \
         PlotRangePadding -> None]"
      ));
      let pts = polyline_points(&svg);
      assert!(close(pts[0].0, 15.0, 1.5), "{cap}: {pts:?}");
      assert!(close(pts[1].0, 345.0, 1.5), "{cap}: {pts:?}");
    }
  }

  /// With a head at position 0 the shaft runs on back through the head
  /// (by the first segment's length, as Wolfram draws it), so the line
  /// shows in the head's notch instead of starting at its tip.
  #[test]
  fn arrowhead_at_start_overlaps_the_shaft() {
    let svg = svg_of(
      "Graphics[{Arrowheads[{{0.05, 0}}], Arrow[{{0, 0}, {3, 0}}]}, \
       PlotRange -> {{-1, 4}, {-1, 1}}, ImageSize -> 360]",
    );
    let pts = polyline_points(&svg);
    assert_eq!(pts.len(), 3, "{pts:?}");
    assert!(close(pts[0].0, -144.0, 1.0), "{pts:?}");
    assert!(close(pts[1].0, 72.0, 1.0), "{pts:?}");
    assert!(close(pts[2].0, 288.0, 1.0), "{pts:?}");
  }

  /// A `VertexColors` polygon star-shaped around its first vertex is
  /// blended over the fan from that vertex, as Wolfram does: the shaded
  /// bulb of `ThermometerGauge` (a highlight point inside a circle) was
  /// ear-clipped instead, smearing the rim colors across its middle.
  #[test]
  fn vertex_colors_fan_from_interior_first_vertex() {
    let svg = svg_of(
      "Graphics[Polygon[Join[{{0, 0}}, Table[{Cos[t], Sin[t]}, \
       {t, 0, 2 Pi, 2 Pi/40}]], VertexColors -> Join[{White}, \
       Table[Black, 41]]], ImageSize -> 200]",
    );
    // Every fan triangle has the white center as a corner: each is one
    // gradient from white to the black rim, none a flat rim color.
    assert!(!svg.contains("fill=\"rgb(0,0,0)\""), "black facet");
    assert_eq!(svg.matches("<polygon").count(), 40, "{svg}");
    assert_eq!(
      svg.matches("<linearGradient").count(),
      svg.matches("stop-color=\"#ffffff\"").count(),
      "{svg}"
    );
  }

  /// The cells of a small `Raster` abut at fractional pixel positions; each
  /// anti-aliased on its own, they let the background show through every
  /// shared edge as a grid of light seams. They are drawn with crisp edges.
  #[test]
  fn raster_cells_tile_without_seams() {
    let svg = svg_of(
      "Graphics[Raster[Table[{x, y, 0}, {y, 0, 1, 1/31}, {x, 0, 1, 1/31}]], \
       ImageSize -> 120]",
    );
    let group = svg
      .split("<g shape-rendering=\"crispEdges\">")
      .nth(1)
      .and_then(|rest| rest.split("</g>").next())
      .unwrap_or_else(|| panic!("no crisp-edged group: {svg}"));
    assert_eq!(group.matches("<rect ").count(), 32 * 32, "{svg}");
  }

  /// A `VertexColors` triangle whose corners are shades of one color (a
  /// lit bulb, as `ThermometerGauge` draws) is filled with one linear
  /// gradient, which blends it exactly; small flat facets showed their
  /// steps as bands. Corners of unrelated colors still need the facets.
  #[test]
  fn vertex_colors_shades_of_one_color_fill_with_a_gradient() {
    let shaded = svg_of(
      "Graphics[Polygon[{{0, 0}, {1, 0}, {0, 1}}, VertexColors -> \
       {Hue[0, 0.7, 1], Hue[0, 0.7, 0.3], Hue[0, 0.7, 0.6]}], \
       ImageSize -> 300]",
    );
    assert_eq!(shaded.matches("<polygon").count(), 1, "{shaded}");
    assert_eq!(shaded.matches("<linearGradient").count(), 1, "{shaded}");
    // Opaque: the triangle is outlined in its own gradient, so it leaves
    // no anti-aliasing seam against a neighbor.
    assert!(shaded.contains("stroke=\"url(#"), "{shaded}");
    let mixed = svg_of(
      "Graphics[Polygon[{{0, 0}, {1, 0}, {0, 1}}, VertexColors -> \
       {Red, Green, Blue}], ImageSize -> 300]",
    );
    assert!(mixed.matches("<polygon").count() > 1, "{mixed}");
    assert!(!mixed.contains("<linearGradient"), "{mixed}");
  }

  /// `MeshCellLabel -> {0 -> …}` marks each labeled vertex with its point
  /// (in the default mesh color), as Wolfram draws it.
  #[test]
  fn mesh_vertex_labels_draw_their_points() {
    let out = interpret(
      "Count[Show[MeshRegion[{{0, 0}, {2, 0}, {1, 1}}, Triangle[{1, 2, 3}], \
       MeshCellLabel -> {0 -> \"Index\"}]][[1]], _Point, Infinity]",
    )
    .unwrap();
    assert_eq!(out, "3");
  }
}
