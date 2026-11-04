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
    // Tick labels take their edge's colour.
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
    // Labelled marks 1.5% of the geometric mean of the frame's sides,
    // unlabelled ones three fifths of that.
    let major = lens.iter().cloned().fold(0.0, f64::max);
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
  fn axes_labels_take_the_axes_style_colour() {
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
}
