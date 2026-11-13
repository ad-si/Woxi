use super::*;

fn svg(expr: &str) -> String {
  clear_state();
  interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
}

/// How many filled `<polygon>` elements an SVG draws.
fn polygon_count(svg: &str) -> usize {
  svg.matches("<polygon").count()
}

/// The distinct `fill="rgb(…)"` colors an SVG uses.
fn fills(svg: &str) -> std::collections::BTreeSet<String> {
  svg
    .split("fill=\"rgb(")
    .skip(1)
    .filter_map(|s| s.split(')').next().map(str::to_string))
    .collect()
}

mod solids {
  use super::*;

  #[test]
  fn scaled_polygon_coordinates_use_the_scene_bounds() {
    assert_eq!(
      svg(
        "Graphics3D[Polygon[{Scaled[{0, 0, 0}], Scaled[{1, 0, 0}], Scaled[{0, 1, 1}]}], PlotRange -> {{2, 4}, {3, 7}, {-2, 2}}, Lighting -> None]"
      ),
      svg(
        "Graphics3D[Polygon[{{2, 3, -2}, {4, 3, -2}, {2, 7, 2}}], PlotRange -> {{2, 4}, {3, 7}, {-2, 2}}, Lighting -> None]"
      )
    );
  }

  #[test]
  fn uniform_vertex_opacity_is_applied_to_the_polygon() {
    assert_eq!(
      svg(
        "Graphics3D[{EdgeForm[], Polygon[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}, VertexColors -> {Directive[Opacity[0.25], Red], Directive[Opacity[0.25], Green], Directive[Opacity[0.25], Blue]}]}, Lighting -> None]"
      ),
      svg(
        "Graphics3D[{EdgeForm[], Opacity[0.25], Polygon[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}, VertexColors -> {Red, Green, Blue}]}, Lighting -> None]"
      )
    );
  }

  #[test]
  fn ellipsoid_with_semi_axes_is_drawn() {
    let s = svg("Graphics3D[Ellipsoid[{0, 0, 0}, {2, 1, 0.5}]]");
    assert!(polygon_count(&s) > 100, "got: {s}");
  }

  #[test]
  fn ellipsoid_with_covariance_matrix_is_drawn() {
    let s = svg(
      "Graphics3D[Ellipsoid[{0, 0, 0}, {{4, 1, 0}, {1, 1, 0}, {0, 0, 0.25}}]]",
    );
    assert!(polygon_count(&s) > 100, "got: {s}");
  }

  // An ellipsoid with semi-axes {a, b, c} is the unit sphere scaled per
  // axis, so both pictures are the same.
  #[test]
  fn ellipsoid_matches_scaled_sphere() {
    assert_eq!(
      svg("Graphics3D[Ellipsoid[{1, 2, 3}, {2, 1, 0.5}]]"),
      svg(
        "Graphics3D[Translate[Scale[Sphere[{0, 0, 0}, 1], {2, 1, 0.5}, \
         {0, 0, 0}], {1, 2, 3}]]"
      )
    );
  }

  #[test]
  fn parallelepiped_draws_its_six_faces() {
    let s = svg(
      "Graphics3D[Parallelepiped[{0, 0, 0}, {{1, 0, 0}, {0.5, 1, 0}, \
       {0, 0.3, 1}}]]",
    );
    assert!(polygon_count(&s) >= 6, "got: {s}");
  }

  #[test]
  fn parallelepiped_is_the_matching_hexahedron() {
    assert_eq!(
      svg(
        "Graphics3D[Parallelepiped[{0, 0, 0}, {{1, 0, 0}, {0.5, 1, 0}, \
         {0, 0.3, 1}}]]"
      ),
      svg(
        "Graphics3D[Hexahedron[{{0, 0, 0}, {1, 0, 0}, {1.5, 1, 0}, \
         {0.5, 1, 0}, {0, 0.3, 1}, {1, 0.3, 1}, {1.5, 1.3, 1}, \
         {0.5, 1.3, 1}}]]"
      )
    );
  }

  #[test]
  fn simplex_of_four_points_is_a_tetrahedron() {
    assert_eq!(
      svg("Graphics3D[Simplex[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]"),
      svg(
        "Graphics3D[Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, \
         {0, 0, 1}}]]"
      )
    );
  }

  #[test]
  fn simplex_of_three_points_is_a_triangle() {
    assert_eq!(
      svg("Graphics3D[Simplex[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}]]"),
      svg("Graphics3D[Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}]]")
    );
  }

  #[test]
  fn simplex_of_two_points_is_a_line() {
    assert_eq!(
      svg("Graphics3D[Simplex[{{0, 0, 0}, {1, 1, 1}}]]"),
      svg("Graphics3D[Line[{{0, 0, 0}, {1, 1, 1}}]]")
    );
  }

  #[test]
  fn capsule_shape_is_drawn() {
    let s = svg("Graphics3D[CapsuleShape[{{0, 0, 0}, {1, 1, 1}}, 0.3]]");
    assert!(polygon_count(&s) > 100, "got: {s}");
  }

  #[test]
  fn arrow_tube_is_drawn() {
    let s = svg(
      "Graphics3D[{Orange, Arrow[Tube[{{0, 0, 0}, {1, 1, 0}, {1, 1, 1}}, \
       0.05]]}]",
    );
    assert!(polygon_count(&s) > 100);
  }
}

mod vertex_colors {
  use super::*;

  // The corner colors are blended across the face instead of the face
  // being drawn in one flat color.
  #[test]
  fn polygon_vertex_colors_blend_across_the_face() {
    let s = svg(
      "Graphics3D[Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 1}, {0, 1, 1}}, \
       VertexColors -> {Red, Green, Blue, Yellow}]]",
    );
    // Independent RGB directions require a two-dimensional color field.
    use base64::Engine;
    let colors: Vec<_> = s
      .split("data:image/png;base64,")
      .skip(1)
      .flat_map(|rest| {
        let bytes = base64::engine::general_purpose::STANDARD
          .decode(rest.split('"').next().unwrap())
          .unwrap();
        ::image::load_from_memory(&bytes)
          .unwrap()
          .to_rgb8()
          .pixels()
          .map(|p| (p[0], p[1], p[2]))
          .collect::<Vec<_>>()
      })
      .collect();
    assert!(!colors.is_empty(), "missing two-dimensional vertex colors");
    // Near the red corner the face is (lit) red: no green or blue.
    assert!(
      colors.iter().any(|&(r, g, b)| r > 150 && g < 60 && b < 60),
      "got: {colors:?}"
    );
  }

  #[test]
  fn line_vertex_colors_color_each_segment() {
    let s = svg(
      "Graphics3D[Line[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}, \
       VertexColors -> {Red, Red, Blue}]]",
    );
    assert!(s.contains("stroke=\"rgb(255,0,0)\""), "got: {s}");
    assert!(s.contains("stroke=\"rgb(128,0,128)\""), "got: {s}");
  }
}

mod list_contour_plot3d {
  use super::*;

  // `array[[i, j, k]]` sits at {k - 1, j - 1, i - 1}: the level 2.5 of a
  // field growing with the first index is the plane z = 1.5.
  #[test]
  fn data_index_order_and_origin() {
    clear_state();
    assert_eq!(
      interpret(
        "g = ListContourPlot3D[Table[x, {x, 1, 4}, {y, 1, 3}, {z, 1, 5}], \
         Contours -> {2.5}, Mesh -> None]; \
         {Head[g], Union[Round[Cases[g, GraphicsComplex[p_, ___] :> p, \
         Infinity][[1]][[All, 3]], 0.001]], MinMax[Cases[g, \
         GraphicsComplex[p_, ___] :> p, Infinity][[1]][[All, 1]]]}"
      )
      .unwrap(),
      "{Graphics3D, {1.5}, {0., 4.}}"
    );
  }

  #[test]
  fn draws_the_sphere_level() {
    let s = svg(
      "ListContourPlot3D[Table[x^2 + y^2 + z^2, {x, -1, 1, 0.25}, \
       {y, -1, 1, 0.25}, {z, -1, 1, 0.25}], Contours -> {0.5}, Mesh -> None]",
    );
    assert!(polygon_count(&s) > 100, "got: {s}");
    // Without a DataRange the box is fitted to the surface, whose ticks
    // run 2, 4, 6 (in sample-index units).
    assert!(s.contains(">6</text>"), "got: {s}");
  }

  #[test]
  fn data_range_spans_the_grid() {
    clear_state();
    assert_eq!(
      interpret(
        "g = ListContourPlot3D[Table[x, {x, 1, 4}, {y, 1, 3}, {z, 1, 5}], \
         Contours -> {2.5}, DataRange -> {{0, 1}, {0, 1}, {0, 1}}]; \
         Union[Round[Cases[g, GraphicsComplex[p_, ___] :> p, \
         Infinity][[1]][[All, 3]], 0.001]]"
      )
      .unwrap(),
      "{0.5}"
    );
  }
}

mod contour_plot3d {
  use super::*;

  #[test]
  fn default_box_ratios_are_a_cube() {
    clear_state();
    assert_eq!(
      interpret(
        "BoxRatios /. Options[ContourPlot3D[x^2 + y^2 + z^2 == 1, \
         {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]]"
      )
      .unwrap(),
      "{1, 1, 1}"
    );
  }
}

mod stream_plot3d {
  use super::*;

  #[test]
  fn draws_colored_streamlines() {
    clear_state();
    assert_eq!(
      interpret(
        "g = StreamPlot3D[{-y, x, 0.2}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
         StreamPoints -> 10]; {Head[g], Length[Cases[g, _Line, Infinity]], \
         FreeQ[Cases[g, _Line, Infinity], Line[_]]}"
      )
      .unwrap(),
      "{Graphics3D, 10, True}"
    );
  }

  // The lattice seeds sit 10% inside the box, as in wolframscript.
  #[test]
  fn corner_seed_lies_on_its_streamline() {
    clear_state();
    assert_eq!(
      interpret(
        "g = StreamPlot3D[{-y, x, 0.2}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
         StreamPoints -> 8]; Min[Norm[# - {-0.8, -0.8, -0.8}] & /@ \
         Flatten[Cases[g, Line[p_, ___] :> p, Infinity], 1]] < 10^-9"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn renders_svg() {
    let s = svg(
      "StreamPlot3D[{-y, x, 0.2}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       StreamPoints -> 10]",
    );
    assert!(s.matches("<line").count() > 100, "got: {s}");
  }
}

/// The `width`/`height` the root `<svg>` element declares.
fn size(svg: &str) -> (f64, f64) {
  let header = &svg[..svg.find('>').unwrap()];
  let attr = |name: &str| -> f64 {
    let key = format!(" {name}=\"");
    let rest = &header[header.find(&key).unwrap() + key.len()..];
    rest[..rest.find('"').unwrap()].parse().unwrap()
  };
  (attr("width"), attr("height"))
}

mod framing {
  use super::*;

  /// The x range of the drawn `<line>` endpoints in an SVG.
  fn line_x_extent(svg: &str) -> (f64, f64) {
    let mut xs = Vec::new();
    for line in svg.lines().filter(|l| l.starts_with("<line ")) {
      for key in ["x1=\"", "x2=\""] {
        if let Some(rest) = line.split(key).nth(1)
          && let Some(v) = rest.split('"').next()
          && let Ok(x) = v.parse::<f64>()
        {
          xs.push(x);
        }
      }
    }
    xs.iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
        (lo.min(x), hi.max(x))
      })
  }

  // `PlotRangePadding` widens the displayed box around an explicit
  // `PlotRange`; a negative padding narrows it and zooms in. Seen from
  // straight above, a square of half-width 7 in a range of half-width 10
  // spans 7/(10 + p) of the picture's half-width.
  #[test]
  fn plot_range_padding_scales_the_framing() {
    let span = |pad: &str| {
      let s = svg(&format!(
        "Graphics3D[{{EdgeForm[Black], FaceForm[None], \
         Cuboid[{{-7, -7, 0}}, {{7, 7, 1}}]}}, \
         PlotRange -> {{{{-10, 10}}, {{-10, 10}}, {{0, 2}}}}, \
         Boxed -> False, ViewPoint -> {{0, 0, 100}}, ImageSize -> 400, \
         PlotRangePadding -> {pad}]"
      ));
      let (lo, hi) = line_x_extent(&s);
      (hi - lo) / 400.0
    };
    let close = |a: f64, b: f64| (a - b).abs() < 0.01;
    assert!(close(span("None"), 0.7), "{}", span("None"));
    assert!(close(span("-2"), 7.0 / 8.0), "{}", span("-2"));
    assert!(close(span("2"), 7.0 / 12.0), "{}", span("2"));
    assert!(
      close(span("Scaled[0.1]"), 7.0 / 12.0),
      "{}",
      span("Scaled[0.1]")
    );
    assert!(
      close(span("{{2, 2}, 0, 0}"), 14.0 / 24.0),
      "{}",
      span("{{2, 2}, 0, 0}")
    );
  }

  // wolframscript draws the unit cube 360 x 391: the height follows the
  // box projected in perspective from the default ViewPoint.
  #[test]
  fn picture_height_follows_the_projected_box() {
    let (w, h) = size(&svg("Graphics3D[Cuboid[]]"));
    assert_eq!(w, 360.0);
    assert!((384.0..=395.0).contains(&h), "height {h}");
    // A flat wide box gives a wide picture (wolframscript: 360 x 237 for
    // this ellipsoid).
    let (_, h) = size(&svg("Graphics3D[Ellipsoid[{0, 0, 0}, {2, 1, 0.5}]]"));
    assert!((225.0..=250.0).contains(&h), "height {h}");
    // Plot3D's 1 x 1 x 0.4 box with its tick labels (wolframscript 290.7;
    // Woxi's labels sit a little closer to the box).
    let (_, h) = size(&svg("Plot3D[Sin[x] Cos[y], {x, -3, 3}, {y, -3, 3}]"));
    assert!((265.0..=295.0).contains(&h), "height {h}");
  }

  #[test]
  fn explicit_image_size_pair_keeps_its_height() {
    assert_eq!(
      size(&svg("Graphics3D[Cuboid[], ImageSize -> {300, 200}]")),
      (300.0, 200.0)
    );
  }

  // In perspective the cube's near vertical edge is drawn longer than the
  // far ones; orthographically (a ViewPoint at infinity) they are equal.
  #[test]
  fn default_view_is_perspective() {
    let pic = |vp: &str| {
      svg(&format!(
        "Graphics3D[{{EdgeForm[], Cuboid[]}}, ViewPoint -> {vp}, \
         Boxed -> False]"
      ))
    };
    assert_ne!(pic("{1.3, -2.4, 2}"), pic("{1.3, -2.4, 2} Infinity"));
  }

  // `ViewPoint -> Top` is the finite `Above` (wolframscript draws the
  // box's far side inset), not an orthographic view.
  #[test]
  fn top_is_above() {
    let pic =
      |vp: &str| svg(&format!("Graphics3D[Cuboid[], ViewPoint -> {vp}]"));
    assert_eq!(pic("Top"), pic("Above"));
    assert_eq!(pic("Bottom"), pic("Below"));
  }

  // A cone's base is a disk: its box is 2 x 2 x 2 (a cube), not stretched
  // by the radius along the axis.
  #[test]
  fn cone_bounds_are_its_disk_and_apex() {
    assert_eq!(
      size(&svg("Graphics3D[Cone[{{0, 0, 0}, {0, 0, 2}}, 1]]")),
      size(&svg(
        "Graphics3D[{Opacity[0], Cuboid[{-1, -1, 0}, {1, 1, 2}]}]"
      ))
    );
  }

  // An axis the contents do not extend along gets the range Wolfram gives
  // it: {0, 2c} at c, {-1, 1} at 0.
  #[test]
  fn flat_scene_gets_depth() {
    assert_eq!(
      size(&svg(
        "Graphics3D[Polygon[{{0, 0, 5}, {10, 0, 5}, {10, 10, 5}}]]"
      )),
      size(&svg(
        "Graphics3D[{Polygon[{{0, 0, 5}, {10, 0, 5}, {10, 10, 5}}], \
         Point[{0, 0, 0}], Point[{0, 0, 10}]}]"
      ))
    );
  }
}

mod plot_range_clipping {
  use super::*;

  // `PlotRange -> {…, {0, 1}}` cuts the unit sphere down to its upper
  // half instead of letting it spill below the box.
  #[test]
  fn sphere_is_cut_to_the_range() {
    // Seen from the front, the picture is framed by the box; the cut-away
    // lower half would reach far below it.
    let half = svg(
      "Graphics3D[Sphere[], PlotRange -> {{-1, 1}, {-1, 1}, {0, 1}}, \
       ViewPoint -> Front]",
    );
    assert!(polygon_count(&half) > 50, "{half}");
    let (_, height) = size(&half);
    for poly in half.split("<polygon points=\"").skip(1) {
      let pts = poly.split('"').next().unwrap();
      for pt in pts.split(' ') {
        let y: f64 = pt.split(',').nth(1).unwrap().parse().unwrap();
        assert!(y <= height + 1.0, "spills below the box: {y}");
      }
    }
  }

  // Text and points anchored outside an explicit range are not drawn.
  #[test]
  fn text_and_points_outside_the_range_are_dropped() {
    let clipped = svg(
      "Graphics3D[{Text[\"far\", {0, 0, 5}], Text[\"near\", {0, 0, 0}], \
       Point[{0, 0, 0}], Point[{3, 3, 3}]}, PlotRange -> 1]",
    );
    assert!(clipped.contains("near"), "{clipped}");
    assert!(!clipped.contains("far"), "{clipped}");
    let shown = svg(
      "Graphics3D[{Text[\"far\", {0, 0, 5}], Point[{0, 0, 0}]}, \
       PlotRange -> 6]",
    );
    assert!(shown.contains("far"), "{shown}");
    let one = svg("Graphics3D[{Point[{0, 0, 0}]}, PlotRange -> 1]");
    let two =
      svg("Graphics3D[{Point[{0, 0, 0}], Point[{3, 3, 3}]}, PlotRange -> 1]");
    assert_eq!(one, two);
  }
}

mod outlines {
  use super::*;

  // Cylinder and cone rims are outlined by the default edge, like
  // wolframscript's; `EdgeForm[]` removes them.
  #[test]
  fn rims_are_outlined_by_default() {
    let lines = |s: &str| s.lines().filter(|l| l.starts_with("<line ")).count();
    let plain = svg("Graphics3D[Cylinder[], Boxed -> False]");
    let bare = svg("Graphics3D[{EdgeForm[], Cylinder[]}, Boxed -> False]");
    assert!(lines(&plain) > 10, "{plain}");
    assert_eq!(lines(&bare), 0);
    let cone = svg("Graphics3D[Cone[], Boxed -> False]");
    assert!(lines(&cone) > 10, "{cone}");
  }

  // A large face pierced by a thin cylinder is ordered against it by
  // cutting the cylinder's small facets along the face's plane, not by
  // cutting the face along the plane of every facet near the piercing,
  // which fanned it into slivers whose seams showed.
  #[test]
  fn a_face_pierced_by_a_thin_cylinder_stays_whole() {
    for (cylinder, face) in [
      (
        "Cylinder[{{-0.5, 0, 0.3}, {1.5, 0, 0.3}}, 0.02]",
        "Polygon[{{0, 0, 1}, {1, -0.7, 0}, {1, 0.7, 0}}]",
      ),
      (
        "Cylinder[{{0, 0, -1}, {0, 0, 2}}, 0.03]",
        "Polygon[{{0, 0, 0}, {1, -0.7, 0}, {1, 0.7, 0.5}}]",
      ),
    ] {
      let s = svg(&format!(
        "Graphics3D[{{{cylinder}, {{Black, Glow[Red], {face}}}}}, \
         Boxed -> False, ViewPoint -> {{3, -4, 1}}]"
      ));
      assert_eq!(s.matches("fill=\"rgb(255,0,0)\"").count(), 1, "{face}");
    }
  }

  // The far side of an opaque closed solid is not drawn, so it cannot be
  // painted over its near side.
  #[test]
  fn hidden_facets_of_a_closed_solid_are_dropped() {
    let opaque = polygon_count(&svg("Graphics3D[Sphere[], Boxed -> False]"));
    let glassy = polygon_count(&svg(
      "Graphics3D[{Opacity[0.5], Sphere[]}, Boxed -> False]",
    ));
    assert!(opaque < glassy * 3 / 4, "{opaque} vs {glassy}");
  }

  // A translucent line drawn in pieces joins them with butt caps, so the
  // overlaps do not darken into dots.
  #[test]
  fn translucent_line_pieces_have_butt_caps() {
    let s = svg("Graphics3D[{Opacity[0.5], Line[{{0, 0, 0}, {1, 1, 1}}]}]");
    assert!(s.contains("stroke-linecap=\"butt\""), "{s}");
    let s = svg("Graphics3D[Line[{{0, 0, 0}, {1, 1, 1}}]]");
    assert!(
      s.lines()
        .filter(|l| l.contains("stroke=\"black\""))
        .all(|l| l.contains("stroke-linecap=\"round\""))
    );
  }
}

mod ticks {
  use super::*;

  // Automatic ticks share the decimals their step needs.
  #[test]
  fn tick_labels_share_their_decimals() {
    let s = svg("Plot3D[Sin[x] Cos[y], {x, -3, 3}, {y, -3, 3}]");
    for label in [">-0.5</text>", ">0.0</text>", ">0.5</text>", ">1.0</text>"] {
      assert!(s.contains(label), "missing {label}");
    }
  }

  // `{x, label}` pairs in `Ticks` label the tick with their label.
  #[test]
  fn explicit_tick_labels_are_drawn() {
    let s = svg(
      "Graphics3D[Cuboid[{0, 0, 0}, {2, 2, 2}], Axes -> True, \
       Ticks -> {{0, 1, 2}, {0, 2}, {{0, \"low\"}, {2, \"high\"}}}]",
    );
    assert!(
      s.contains(">low</text>") && s.contains(">high</text>"),
      "{s}"
    );
  }
}

mod plot3d_options {
  use super::*;

  fn lines_with(svg: &str, stroke: &str) -> usize {
    svg
      .lines()
      .filter(|l| l.starts_with("<line ") && l.contains(stroke))
      .count()
  }

  // The surface's edges are outlined in black by default (`BoundaryStyle
  // -> GrayLevel[0]`); `BoundaryStyle` restyles or removes the outline.
  #[test]
  fn boundary_style() {
    let plain = svg("Plot3D[x y, {x, -1, 1}, {y, -1, 1}, Mesh -> None]");
    assert!(
      lines_with(&plain, "stroke=\"rgb(0,0,0)\"") >= 100,
      "{plain}"
    );
    let red = svg(
      "Plot3D[x y, {x, -1, 1}, {y, -1, 1}, Mesh -> None, \
       BoundaryStyle -> Directive[Thick, Red]]",
    );
    assert!(lines_with(&red, "stroke=\"rgb(255,0,0)\"") >= 100);
    let none = svg(
      "Plot3D[x y, {x, -1, 1}, {y, -1, 1}, Mesh -> None, \
       BoundaryStyle -> None]",
    );
    assert_eq!(lines_with(&none, "stroke=\"rgb(0,0,0)\""), 0);
  }

  // `Mesh -> n` draws n lines per direction, in the `MeshStyle`.
  #[test]
  fn mesh_count_and_style() {
    let s = svg(
      "Plot3D[Cos[x] Sin[y], {x, -Pi, Pi}, {y, -Pi, Pi}, Mesh -> 5, \
       MeshStyle -> Red, BoundaryStyle -> None]",
    );
    let red = lines_with(&s, "stroke=\"rgb(255,0,0)\"");
    // Ten lines, each crossing the 50 grid cells in about 50 pieces.
    assert!((400..=1200).contains(&red), "{red}");
    assert_eq!(lines_with(&s, "stroke=\"#000000a0\""), 0);
  }

  // `MeshShading` paints the patches between the mesh lines.
  #[test]
  fn mesh_shading_checkerboard() {
    let s = svg(
      "Plot3D[x y, {x, -1, 1}, {y, -1, 1}, Mesh -> 4, \
       MeshShading -> {{Red, Yellow}, {Yellow, Red}}]",
    );
    let colors = fills(&s);
    let reddish = colors.iter().any(|c| {
      let v: Vec<u8> = c
        .split(',')
        .map(|x| x.trim().parse::<u8>().unwrap())
        .collect();
      v[0] > 150 && v[1] < 60
    });
    let yellowish = colors.iter().any(|c| {
      let v: Vec<u8> = c
        .split(',')
        .map(|x| x.trim().parse::<u8>().unwrap())
        .collect();
      v[0] > 150 && v[1] > 150 && v[2] < 60
    });
    assert!(reddish && yellowish, "{colors:?}");
  }

  // `RegionFunction` keeps only the part of the surface it accepts, and
  // the vertical range is that of what remains.
  #[test]
  fn region_function_cuts_the_surface() {
    let s = svg(
      "Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}, \
       RegionFunction -> (#1^2 + #2^2 <= 4 &)]",
    );
    assert!(s.contains(">4</text>") && !s.contains(">8</text>"), "{s}");
  }

  #[test]
  fn mesh_functions_draw_their_contours() {
    let rings = svg(
      "Plot3D[Exp[-(x^2 + y^2)], {x, -2, 2}, {y, -2, 2}, \
       MeshFunctions -> {#1^2 + #2^2 &}, Mesh -> 6, BoundaryStyle -> None]",
    );
    let grid = svg(
      "Plot3D[Exp[-(x^2 + y^2)], {x, -2, 2}, {y, -2, 2}, Mesh -> 6, \
       BoundaryStyle -> None]",
    );
    assert_ne!(rings, grid);
    assert!(lines_with(&rings, "stroke=\"#000000a0\"") > 100);
  }

  #[test]
  fn filling_hangs_a_wall_to_the_bottom() {
    let s = svg("Plot3D[Sin[x y], {x, 0, 3}, {y, 0, 3}, Filling -> Bottom]");
    assert!(s.contains("fill=\"rgb(178,178,178)\""), "{s}");
    let s = svg("Plot3D[Sin[x y], {x, 0, 3}, {y, 0, 3}]");
    assert!(!s.contains("fill=\"rgb(178,178,178)\""));
  }

  #[test]
  fn plot_legends_expressions() {
    let s = svg(
      "Plot3D[{x + y, x - y}, {x, -1, 1}, {y, -1, 1}, \
       PlotLegends -> \"Expressions\"]",
    );
    assert!(
      s.contains(">x + y</text>") && s.contains(">x - y</text>"),
      "{s}"
    );
  }
}

mod plot3d_family {
  use super::*;

  // `PlotStyle -> Thick` thickens a space curve without dropping its
  // default color (the first plot color, not black).
  #[test]
  fn thick_space_curve_keeps_its_color() {
    let s = svg(
      "ParametricPlot3D[{Cos[t], Sin[t], t/4}, {t, 0, 6 Pi}, \
       PlotStyle -> Thick]",
    );
    assert!(!s.contains("stroke=\"black\""), "{s}");
    assert!(!s.contains("stroke=\"rgb(0,0,0)\""));
  }

  // Scattered `{x, y, z}` points are triangulated, not read as a height
  // matrix of their coordinates: the axes span the points' x and y.
  #[test]
  fn list_plot3d_scattered_points() {
    clear_state();
    let s = svg(
      "SeedRandom[1]; pts = RandomReal[{0, 3}, {80, 2}]; \
       ListPlot3D[Map[Append[#, Sin[#[[1]] #[[2]]]] &, pts]]",
    );
    assert!(polygon_count(&s) > 500, "{s}");
    assert!(!s.contains(">80</text>") && !s.contains(">60</text>"));
  }

  // A parametric surface carries the default mesh of constant-u and
  // constant-v lines.
  #[test]
  fn parametric_surface_has_a_mesh() {
    let torus = "ParametricPlot3D[{(2 + Cos[v]) Cos[u], (2 + Cos[v]) Sin[u], \
                 Sin[v]}, {u, 0, 2 Pi}, {v, 0, 2 Pi}";
    let meshed = svg(&format!("{torus}]"));
    let bare = svg(&format!("{torus}, Mesh -> None]"));
    let count = |s: &str| s.matches("stroke=\"#000000a0\"").count();
    assert!(count(&meshed) > 200, "{}", count(&meshed));
    assert_eq!(count(&bare), 0);
  }
}

mod level_surfaces {
  use super::*;

  // Each contour level takes the next theme color (orange, blue, green),
  // as in wolframscript.
  #[test]
  fn contour_levels_take_theme_colors() {
    let one = fills(&svg(
      "ContourPlot3D[x^2 + y^2 + z^2, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       Contours -> {0.5}, Mesh -> None]",
    ));
    let three = fills(&svg(
      "ContourPlot3D[x^2 + y^2 + z^2, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       Contours -> {0.2, 0.5, 0.9}, Mesh -> None]",
    ));
    // Blue appears only among several levels.
    let bluish = |set: &std::collections::BTreeSet<String>| {
      set.iter().any(|c| {
        let v: Vec<u8> = c
          .split(',')
          .map(|x| x.trim().parse::<u8>().unwrap())
          .collect();
        u16::from(v[2]) > u16::from(v[0]) + 40
      })
    };
    assert!(!bluish(&one) && bluish(&three));
  }

  // A region of inequalities is drawn as a smooth surface, closed where it
  // meets the box, instead of a staircase of voxels.
  #[test]
  fn region_plot3d_is_a_smooth_surface() {
    let s = svg(
      "RegionPlot3D[x^2 + y^2 + z^2 < 1 && (x - 0.8)^2 + y^2 + z^2 < 1, \
       {x, -1, 2}, {y, -1, 1}, {z, -1, 1}]",
    );
    assert!(polygon_count(&s) > 500, "{s}");
    // BoxRatios {1, 1, 1}: as tall as a cube's projection.
    let (w, h) = size(&s);
    assert!(h / w > 1.0, "{w} x {h}");
  }

  #[test]
  fn region_plot3d_keeps_its_axes_labels() {
    let s = svg(
      "RegionPlot3D[x^2 + y^2 < 1, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       AxesLabel -> {\"alpha\", \"beta\", \"gamma\"}]",
    );
    assert!(s.contains("alpha") && s.contains("gamma"));
  }

  #[test]
  fn region_signed_field_shapes() {
    clear_state();
    // The region is closed at the box faces: a cylinder cut by the box
    // has flat caps facing up.
    let s =
      svg("RegionPlot3D[x^2 + y^2 < 1, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]");
    assert!(polygon_count(&s) > 500);
  }
}

mod fields_and_graphs {
  use super::*;

  // Solid arrows (a shaft and a head) centered on the cells of an n³ grid,
  // colored along "WL12DefaultVectorGradient", in a 1 x 1 x 1 box.
  #[test]
  fn vector_plot3d_draws_solid_arrows() {
    clear_state();
    assert_eq!(
      interpret(
        "g = VectorPlot3D[{-y, x, z}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
         VectorPoints -> 4]; {Length[Cases[g, Arrow[_Tube], Infinity]] == Length[Cases[g, _Arrow, Infinity]], \
         BoxRatios /. Options[g]}"
      )
      .unwrap(),
      "{True, {1, 1, 1}}"
    );
    let s = svg(
      "VectorPlot3D[{-y, x, z}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       VectorPoints -> 4]",
    );
    assert!(polygon_count(&s) > 64 * 10, "{s}");
  }

  // The longest streamlines end in the gradient's orange, the slowest in
  // its dark blue.
  #[test]
  fn stream_plot3d_uses_the_vector_gradient() {
    clear_state();
    assert_eq!(
      interpret(
        "c = Join @@ Cases[StreamPlot3D[{-y, x, 0.2}, {x, -1, 1}, \
         {y, -1, 1}, {z, -1, 1}, StreamPoints -> 10], \
         Line[_, VertexColors -> v_] :> v, Infinity]; \
         c = (If[ListQ[#], RGBColor @@ #, ColorConvert[#, \"RGB\"]] &) /@ c; \
         {MemberQ[c, RGBColor[r_, g_, b_] /; b > 0.6 && r < 0.1], \
         MemberQ[c, RGBColor[r_, g_, b_] /; r > 0.95 && b < 0.1]}"
      )
      .unwrap(),
      "{True, True}"
    );
  }

  // A `Graph` is laid out in space: spheres for the vertices and tubes for
  // the edges, without a box.
  #[test]
  fn graph_plot3d_of_a_graph() {
    let s = svg("GraphPlot3D[HypercubeGraph[3]]");
    assert!(polygon_count(&s) > 1000, "{s}");
    let boxed = svg("GraphPlot3D[HypercubeGraph[3], Boxed -> True]");
    assert_ne!(s, boxed);
  }

  #[test]
  fn list_plot3d_mesh_style() {
    let s = svg(
      "ListPlot3D[Table[Cos[i/2] Sin[j/2], {i, 0, 10}, {j, 0, 10}], \
       Mesh -> All, MeshStyle -> Blue]",
    );
    assert!(s.matches("stroke=\"rgb(0,0,255)\"").count() > 100, "{s}");
  }
}

mod arrows_and_box {
  use super::*;

  /// The area of the first filled arrowhead `<polygon>` (the triangle after
  /// a `<polyline>`).
  fn head_area(svg: &str) -> f64 {
    let after = svg.split("<polyline").nth(1).unwrap();
    let pts = after.split("<polygon points=\"").nth(1).unwrap();
    let p: Vec<(f64, f64)> = pts
      .split('"')
      .next()
      .unwrap()
      .split(' ')
      .map(|xy| {
        let mut it = xy.split(',').map(|v| v.parse::<f64>().unwrap());
        (it.next().unwrap(), it.next().unwrap())
      })
      .collect();
    ((p[1].0 - p[0].0) * (p[2].1 - p[0].1)
      - (p[2].0 - p[0].0) * (p[1].1 - p[0].1))
      .abs()
      / 2.0
  }

  // `Arrowheads[s]` sizes the heads of 3D arrows as a fraction of the
  // picture's width.
  #[test]
  fn arrowheads_size_3d_arrow_heads() {
    let small = svg("Graphics3D[{Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    let large =
      svg("Graphics3D[{Arrowheads[0.05], Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    assert!(head_area(&large) > 3.0 * head_area(&small));
  }

  // The bounding box is drawn opaque, in the axis gray.
  #[test]
  fn box_edges_are_opaque() {
    let s = svg("Graphics3D[Cuboid[]]");
    assert!(s.contains("stroke=\"rgb(102,102,102)\" stroke-width=\"0.5\""));
    assert!(!s.contains("opacity=\"0.4\""));
  }
}

mod smooth_shading {
  use super::*;

  // A curved surface is lit at its corners and each facet is filled with
  // the gradient between them; flat faces keep one color.
  #[test]
  fn curved_surfaces_are_filled_with_gradients() {
    for code in [
      "Graphics3D[Sphere[]]",
      "Graphics3D[Cylinder[]]",
      "Graphics3D[Cone[]]",
      "Graphics3D[CapsuleShape[{{0, 0, 0}, {1, 1, 1}}, 0.3]]",
      "Plot3D[Sin[x y], {x, 0, 3}, {y, 0, 3}, Mesh -> None]",
    ] {
      let s = svg(code);
      assert!(s.contains("<linearGradient"), "{code}");
      assert!(s.contains("fill=\"url(#"), "{code}");
    }
    let flat = svg("Graphics3D[Cuboid[]]");
    assert!(!flat.contains("<linearGradient"), "{flat}");
  }

  // The far side of a closed convex surface (a capsule) is hidden, so a
  // translucent one, whose far side shows, draws more facets.
  #[test]
  fn far_side_of_a_capsule_is_dropped() {
    let opaque = polygon_count(&svg(
      "Graphics3D[CapsuleShape[{{0, 0, 0}, {1, 1, 1}}, 0.3], Boxed -> False]",
    ));
    let glassy = polygon_count(&svg(
      "Graphics3D[{Opacity[0.5], CapsuleShape[{{0, 0, 0}, {1, 1, 1}}, \
       0.3]}, Boxed -> False]",
    ));
    assert!(glassy > opaque * 3 / 2, "{opaque} vs {glassy}");
  }

  // Of two faces in one plane, the one given later is drawn on top.
  #[test]
  fn coplanar_faces_draw_in_order() {
    let s = svg(
      "Graphics3D[{Red, Polygon[{{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, \
       0}}], Blue, Polygon[{{-1, -1, 0}, {5, -1, 0}, {-1, 5, 0}}]}]",
    );
    let colors: Vec<(u32, u32, u32)> = s
      .split("<polygon ")
      .skip(1)
      .filter_map(|p| {
        let rgb = p.split("fill=\"rgb(").nth(1)?.split(')').next()?;
        let v: Vec<u32> =
          rgb.split(',').filter_map(|c| c.parse().ok()).collect();
        Some((v[0], v[1], v[2]))
      })
      .collect();
    let last_red = colors.iter().rposition(|c| c.0 > c.2).unwrap();
    let first_blue = colors.iter().position(|c| c.2 > c.0).unwrap();
    assert!(last_red < first_blue, "{colors:?}");
  }
}

mod edge_form_3d {
  use super::*;

  // `EdgeForm[Dashed]` adds dashing to the edge style already in force.
  #[test]
  fn dashed_edges_keep_the_earlier_edge_color() {
    let s = svg(
      "Graphics3D[{EdgeForm[{Thick, Red}], Cuboid[{0, 0, 0}], EdgeForm[None], \
       Cuboid[{2, 0, 0}], EdgeForm[Dashed], Cuboid[{4, 0, 0}]}]",
    );
    let dashed: Vec<&str> = s
      .lines()
      .filter(|l| l.starts_with("<line ") && l.contains("stroke-dasharray"))
      .collect();
    assert!(dashed.len() >= 9, "{s}");
    assert!(dashed.iter().all(|l| l.contains("stroke=\"rgb(255,0,0)\"")));
  }

  // `FaceForm[None]` leaves the faces unfilled: only the outlines that
  // `EdgeForm` asks for are drawn.
  #[test]
  fn face_form_none_draws_only_the_edges() {
    let s = svg(
      "Graphics3D[{EdgeForm[Red], FaceForm[None], Cuboid[]}, \
       Boxed -> False]",
    );
    assert_eq!(polygon_count(&s), 0, "{s}");
    assert!(s.contains("<line ") && s.contains("stroke=\"rgb(255,0,0)\""));
    // Lines after it keep their own color and opacity.
    let s = svg(
      "Graphics3D[{FaceForm[None], Cuboid[], Blue, \
       Line[{{0, 0, 0}, {2, 2, 2}}]}, Boxed -> False]",
    );
    assert_eq!(polygon_count(&s), 0, "{s}");
    assert!(s.contains("stroke=\"rgb(0,0,255)\""), "{s}");
  }

  // A later `FaceForm[color]` fills the faces again.
  #[test]
  fn face_form_color_after_none_fills_faces_again() {
    let s = svg(
      "Graphics3D[{FaceForm[None], Cuboid[], FaceForm[Red], \
       Cuboid[{2, 0, 0}]}, Boxed -> False, Lighting -> {{\"Ambient\", White}}]",
    );
    assert!(polygon_count(&s) > 0, "{s}");
    assert!(fills(&s).contains("255,0,0"), "{s}");
  }
}

mod contour_plot3d_mesh {
  use super::*;

  // The default mesh follows the planes of the sampling grid.
  #[test]
  fn default_mesh_is_drawn() {
    let mesh = |opt: &str| {
      svg(&format!(
        "ContourPlot3D[x^2 + y^2 - z^2 == 1, {{x, -2, 2}}, {{y, -2, 2}}, \
         {{z, -2, 2}}, PlotPoints -> 8{opt}]"
      ))
      .matches("stroke=\"#000000a0\"")
      .count()
    };
    assert!(mesh("") > 50);
    assert_eq!(mesh(", Mesh -> None"), 0);
  }
}

mod face_grids {
  use super::*;

  #[test]
  fn face_grids_draw_light_lines() {
    let grid = "stroke=\"rgb(217,217,217)\"";
    assert!(svg("Graphics3D[Sphere[], FaceGrids -> All]").contains(grid));
    assert!(!svg("Graphics3D[Sphere[]]").contains(grid));
    // Only the asked face's lines, and none along a `None` direction.
    let one =
      svg("Graphics3D[Sphere[], FaceGrids -> {{{0, 1, 0}, {None, {0}}}}]");
    assert_eq!(one.matches(grid).count(), 20, "{one}");
  }

  // `{x, ""}` is a tick mark without a number.
  #[test]
  fn empty_tick_labels_are_not_numbered() {
    let s = svg(
      "Graphics3D[Cuboid[], Axes -> True, Ticks -> {{{0.5, \"\"}}, None, \
       None}]",
    );
    assert!(!s.contains(">0.5<"), "{s}");
  }
}

mod charts_3d {
  use super::*;

  #[test]
  fn bubble_chart_3d_draws_spheres() {
    clear_state();
    assert_eq!(
      interpret(
        "g = BubbleChart3D[{{1, 2, 3, 1}, {2, 1, 2, 8}}]; \
         {Head[g], Length[Cases[g, _Sphere, Infinity]]}"
      )
      .unwrap(),
      "{Graphics3D, 2}"
    );
    assert!(
      polygon_count(&svg("BubbleChart3D[{{1, 2, 3, 1}, {2, 1, 2, 8}}]")) > 100
    );
  }

  // The pie's rounded wall is shaded smoothly.
  #[test]
  fn pie_chart_3d_wall_is_smooth() {
    assert!(svg("PieChart3D[{3, 5, 2, 4}]").contains("<linearGradient"));
  }
}

mod scattered_list_plot3d {
  use super::*;

  // Scattered points are drawn on their own triangulation, so the surface
  // reaches each point's height (a resampled grid shaved the peak off).
  #[test]
  fn surface_reaches_the_data_peak() {
    let s = svg(
      "ListPlot3D[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, \
       {0.37, 0.61, 2}}]",
    );
    assert!(s.contains(">2.0<") || s.contains(">2<"), "{s}");
  }
}

#[test]
fn raster_volume_colors_do_not_expose_triangle_lighting() {
  let s = svg("Graphics3D[Raster3D[{{{{1,0,0}}}}],Boxed->False]");
  assert!(s.contains("fill=\"rgb(255,0,0)\""));
  assert!(s.split("<polygon").skip(1).all(|p| {
    p.split('>')
      .next()
      .unwrap()
      .contains("fill=\"rgb(255,0,0)\"")
  }));
}

mod per_axis_options {
  use super::*;

  // `PlotRange -> {{x0, x1}, All, All}` pins only the first axis; the
  // others keep the extent of the contents.
  #[test]
  fn plot_range_mixing_intervals_and_all() {
    for spec in ["{{0, 4}, All, All}", "{{0, 4}, Automatic, Full}"] {
      assert_eq!(
        svg(&format!(
          "Graphics3D[Line[{{{{1, 0, 0}}, {{2, 2, 3}}}}], Axes -> True, \
           PlotRange -> {spec}]"
        )),
        svg(
          "Graphics3D[Line[{{1, 0, 0}, {2, 2, 3}}], Axes -> True, \
           PlotRange -> {{0, 4}, {0, 2}, {0, 3}}]"
        ),
        "{spec}"
      );
    }
  }

  #[test]
  fn plot_range_mixing_intervals_and_all_widens_the_box() {
    // The pinned x interval reaches past the data, so its far tick shows.
    let s = svg(
      "Graphics3D[Line[{{1, 0, 0}, {2, 1, 1}}], Axes -> True, \
       PlotRange -> {{0, 4}, All, All}]",
    );
    assert!(s.contains(">4<"), "{s}");
  }

  // `Axes -> {False, True, True}` draws the y and z axes only.
  #[test]
  fn axes_list_turns_off_single_axes() {
    let all = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> True, \
       AxesLabel -> {\"xx\", \"yy\", \"zz\"}]",
    );
    assert!(all.contains(">4<") && all.contains(">xx<"), "{all}");
    let some = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> {False, True, True}, \
       AxesLabel -> {\"xx\", \"yy\", \"zz\"}]",
    );
    assert!(!some.contains(">4<") && !some.contains(">xx<"), "{some}");
    assert!(some.contains(">yy<") && some.contains(">zz<"), "{some}");
  }

  #[test]
  fn axes_list_of_all_false_draws_no_axes() {
    assert_eq!(
      svg(
        "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> {False, False, False}]"
      ),
      svg("Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> False]")
    );
  }
}

mod box_and_tick_styles {
  use super::*;

  // A lone number asks for equal sides, like `{1, 1, 1}`.
  #[test]
  fn scalar_box_ratios_is_a_cube() {
    assert_eq!(
      svg("Graphics3D[Line[{{0, 0, 0}, {4, 1, 2}}], BoxRatios -> 1]"),
      svg("Graphics3D[Line[{{0, 0, 0}, {4, 1, 2}}], BoxRatios -> {1, 1, 1}]")
    );
    assert_ne!(
      svg("Graphics3D[Line[{{0, 0, 0}, {4, 1, 2}}], BoxRatios -> 1]"),
      svg("Graphics3D[Line[{{0, 0, 0}, {4, 1, 2}}]]")
    );
  }

  // `TicksStyle -> {sx, sy, sz}` colors each axis's ticks and labels.
  #[test]
  fn ticks_style_colors_each_axis() {
    let s = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> True, \
       TicksStyle -> {Red, Directive[14, Blue], None}]",
    );
    assert!(s.contains("stroke=\"rgb(255,0,0)\""), "{s}");
    assert!(s.contains("fill=\"rgb(255,0,0)\""), "{s}");
    assert!(s.contains("fill=\"rgb(0,0,255)\""), "{s}");
  }

  #[test]
  fn single_ticks_style_colors_all_axes() {
    let s = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> True, \
       TicksStyle -> Orange]",
    );
    let orange = "fill=\"rgb(255,128,0)\"";
    assert!(s.matches(orange).count() >= 6, "{s}");
  }

  // Options of a parametric plot carry into `Show`, where a mixed
  // `PlotRange` then widens one axis.
  #[test]
  fn parametric_plot_options_survive_show() {
    let s = svg(
      "Show[ParametricPlot3D[{t, Sin[t], Cos[t]}, {t, 0, 4}, \
       TicksStyle -> {Red, Blue, Green}, BoxRatios -> 1], \
       PlotRange -> {{0, 8}, All, All}]",
    );
    assert!(
      s.contains("fill=\"rgb(255,0,0)\"") && s.contains(">8<"),
      "{s}"
    );
  }
}

mod box_and_label_options {
  use super::*;

  const SURFACE: &str = "Plot3D[x y, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 5, \
     Mesh -> None";

  #[test]
  fn plot3d_draws_its_epilog() {
    let s = svg(&format!(
      "{SURFACE}, Epilog -> Inset[Text[\"surface title\"], {{Center, Top}}, \
       {{Center, Top}}]]"
    ));
    assert!(s.contains("surface title"));
  }

  #[test]
  fn plot3d_draws_its_prolog() {
    let s = svg(&format!(
      "{SURFACE}, Prolog -> Text[\"backdrop\", Scaled[{{0.5, 0.5}}]]]"
    ));
    assert!(s.contains("backdrop"));
  }

  #[test]
  fn dashed_box_style_dashes_the_box_edges() {
    for expr in [
      format!("{SURFACE}, BoxStyle -> {{Black, Dashed}}]"),
      "Graphics3D[Cuboid[], BoxStyle -> Directive[Red, Dashed]]".to_string(),
    ] {
      assert!(
        svg(&expr).contains("stroke-dasharray"),
        "no dashed box edges for {expr}"
      );
    }
    assert!(!svg(&format!("{SURFACE}]")).contains("stroke-dasharray"));
  }

  #[test]
  fn box_style_color_and_thickness_apply_to_the_box_edges() {
    let s = svg(&format!(
      "{SURFACE}, BoxStyle -> {{Red, AbsoluteThickness[2]}}]"
    ));
    assert!(s.contains("stroke=\"rgb(255,0,0)\" stroke-width=\"2"));
    let s = svg("Graphics3D[Cuboid[], BoxStyle -> Blue]");
    assert!(s.contains("stroke=\"rgb(0,0,255)\""));
  }

  #[test]
  fn label_style_font_size_sizes_the_tick_labels() {
    for expr in [
      format!("{SURFACE}, LabelStyle -> 16]"),
      format!("{SURFACE}, LabelStyle -> {{FontSize -> 16, Bold}}]"),
      "Graphics3D[Cuboid[], Axes -> True, LabelStyle -> 16]".to_string(),
    ] {
      let s = svg(&expr);
      assert!(s.contains("font-size=\"16\""), "no 16pt labels for {expr}");
      assert!(!s.contains("font-size=\"10\""), "10pt ticks left in {expr}");
    }
  }
}
