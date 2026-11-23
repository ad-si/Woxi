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

  /// A solid wholly inside an opaque sphere or capped cylinder can never be
  /// seen, so it draws nothing. Regression: a two-tone ball (a sphere just
  /// inside another, offset a little) showed the inner sphere's facet
  /// seams streaking across the outer one where the painter's sort
  /// interleaved them.
  #[test]
  fn solid_inside_an_opaque_solid_draws_nothing() {
    let outer = svg("Graphics3D[{Sphere[{0, 0, 0}, 1]}, Boxed -> False]");
    let nested = svg(
      "Graphics3D[{Sphere[{0, 0, 0}, 1], Red, Sphere[{0, 0, 0.1}, 0.5], \
       Cylinder[{{0, -0.3, 0}, {0, 0.3, 0}}, 0.2]}, Boxed -> False]",
    );
    assert_eq!(polygon_count(&nested), polygon_count(&outer));

    let cylinder =
      svg("Graphics3D[{Cylinder[{{0, -1, 0}, {0, 1, 0}}, 1]}, Boxed -> False]");
    let in_cylinder = svg(
      "Graphics3D[{Cylinder[{{0, -1, 0}, {0, 1, 0}}, 1], Red, \
       Sphere[{0, 0.2, 0}, 0.5]}, Boxed -> False]",
    );
    assert_eq!(polygon_count(&in_cylinder), polygon_count(&cylinder));
  }

  /// How jagged the outline of an SVG's polygons is: the longest edge of
  /// the convex hull of every point they draw, relative to the hull's mean
  /// radius. A regular n-gon scores `2 sin(pi/n)`.
  fn outline_roughness(svg: &str) -> f64 {
    let mut pts: Vec<(f64, f64)> = svg
      .split("<polygon points=\"")
      .skip(1)
      .filter_map(|s| s.split('"').next())
      .flat_map(|s| s.split_whitespace())
      .filter_map(|p| {
        let (x, y) = p.split_once(',')?;
        Some((x.parse().ok()?, y.parse().ok()?))
      })
      .collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
      (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    // Andrew's monotone chain: lower hull, then upper hull.
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for pass in 0..2 {
      let start = hull.len();
      let seq: Vec<(f64, f64)> = if pass == 0 {
        pts.clone()
      } else {
        pts.iter().rev().copied().collect()
      };
      for p in seq {
        while hull.len() >= start + 2
          && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
          hull.pop();
        }
        hull.push(p);
      }
      hull.pop();
    }
    let n = hull.len() as f64;
    let cx = hull.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = hull.iter().map(|p| p.1).sum::<f64>() / n;
    let r = hull
      .iter()
      .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
      .sum::<f64>()
      / n;
    let longest = (0..hull.len())
      .map(|i| {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
      })
      .fold(0.0, f64::max);
    longest / r
  }

  /// An opaque sphere is tessellated about the line of sight, so its
  /// outline is its finely divided equator from any view direction — in a
  /// lone-sphere scene and in one with many spheres. Regression: in a
  /// scene of a few hundred spheres the larger ones came out as visibly
  /// jagged 12-gons with saw-toothed intersections, and a lone sphere's
  /// outline was a slanted cut through its latitude bands.
  #[test]
  fn opaque_sphere_outline_is_round_from_any_view() {
    for view in ["{1.3, -2.4, 2}", "{0, 0, 10}", "{3, -1, 0.5}"] {
      let lone = svg(&format!(
        "Graphics3D[Sphere[{{0, 0, 0}}, 1], Boxed -> False, \
         ViewPoint -> {view}]"
      ));
      let rough = outline_roughness(&lone);
      // A 48-gon scores 0.131; the slanted cut scored 0.18.
      assert!(
        rough < 0.15,
        "{view}: lone sphere outline roughness {rough}"
      );
    }
    // Many spheres: small ones floating in front of a large one stay
    // inside its outline, which is still round.
    let crowd = svg(
      "Graphics3D[{Sphere[{0, 0, 0}, 1], Table[Sphere[1.3 {0.94, -0.31, \
       0.16} + 0.04 {Mod[i, 7] - 3, Mod[i, 5] - 2, Mod[i, 3] - 1}, 0.05], \
       {i, 40}]}, Boxed -> False, ViewPoint -> {3, -1, 0.5}]",
    );
    let rough = outline_roughness(&crowd);
    // A 36-gon scores 0.174.
    assert!(rough < 0.2, "crowded sphere outline roughness {rough}");
  }

  /// Only the part of a sphere poking out of another stays; a see-through
  /// enclosing solid hides nothing.
  #[test]
  fn solid_culling_keeps_visible_and_see_through_parts() {
    let outer = svg("Graphics3D[{Sphere[{0, 0, 0}, 1]}, Boxed -> False]");
    let poking = svg(
      "Graphics3D[{Sphere[{0, 0, 0}, 1], Red, Sphere[{0, 0, 0.6}, 0.5]}, \
       Boxed -> False]",
    );
    assert!(polygon_count(&poking) > polygon_count(&outer));
    assert!(fills(&poking).iter().any(|f| {
      let c: Vec<u8> = f.split(',').filter_map(|v| v.parse().ok()).collect();
      c.len() == 3 && c[0] > 2 * c[1].max(c[2]).max(1)
    }));

    let glass =
      svg("Graphics3D[{Opacity[0.5], Sphere[{0, 0, 0}, 1]}, Boxed -> False]");
    let glass_nested = svg(
      "Graphics3D[{Opacity[0.5], Sphere[{0, 0, 0}, 1], Opacity[1], Red, \
       Sphere[{0, 0, 0}, 0.5]}, Boxed -> False]",
    );
    assert!(polygon_count(&glass_nested) > polygon_count(&glass));
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

  /// The solid head of `Arrow[Tube[…]]` is sized by the scene's box (the
  /// diagonal of its `PlotRange`), not by the arrow itself: a short axis
  /// arrow in a large scene keeps a visible head.
  #[test]
  fn arrow_tube_head_is_sized_by_the_scene() {
    let d = 0.05 * 3f64.sqrt() * 4.0;
    let range = "PlotRange -> {{-2, 2}, {-2, 2}, {-2, 2}}";
    assert_eq!(
      svg(&format!(
        "Graphics3D[{{Arrow[Tube[{{{{0, 0, 0}}, {{1, 0, 0}}}}, 0.01]]}}, {range}]"
      )),
      svg(&format!(
        "Graphics3D[{{Tube[{{{{0, 0, 0}}, {{{b}, 0, 0}}}}, 0.01], \
         Cone[{{{{{b}, 0, 0}}, {{1, 0, 0}}}}, {r}]}}, {range}]",
        b = 1.0 - d,
        r = 0.22 * d
      ))
    );
  }

  /// The head covers the last stretch of the curve measured along it, so a
  /// finely sampled curve (an arc) still gets a full-size head instead of
  /// one clipped to its last tiny segment.
  #[test]
  fn arrow_tube_head_spans_several_segments() {
    let len = 0.1 * 3f64.sqrt() * 4.0;
    let range = "PlotRange -> {{-2, 2}, {-2, 2}, {-2, 2}}";
    assert_eq!(
      svg(&format!(
        "Graphics3D[{{Arrowheads[0.1], Arrow[Tube[{{{{0, 0, 0}}, \
         {{0.5, 0, 0}}, {{0.75, 0, 0}}, {{1, 0, 0}}}}, 0.01]]}}, {range}]"
      )),
      svg(&format!(
        "Graphics3D[{{Tube[{{{{0, 0, 0}}, {{{b}, 0, 0}}}}, 0.01], \
         Cone[{{{{{b}, 0, 0}}, {{1, 0, 0}}}}, {r}]}}, {range}]",
        b = 1.0 - len,
        r = 0.22 * len
      ))
    );
  }

  /// `Style[Text[…], Italic]` (the style outside the `Text`) slants the
  /// label, as `Style[…, Bold]` makes it bold.
  #[test]
  fn styled_text_label_font_face() {
    let s = svg("Graphics3D[{Style[Text[\"X\", {0, 0, 0}], 17, Italic]}]");
    assert!(s.contains("font-style=\"italic\""), "got: {s}");
    let s = svg("Graphics3D[{Style[Text[\"X\", {0, 0, 0}], Bold]}]");
    assert!(s.contains("font-weight=\"bold\""), "got: {s}");
    let s =
      svg("Graphics3D[{Style[Text[\"X\", {0, 0, 0}], FontSlant -> Italic]}]");
    assert!(s.contains("font-style=\"italic\""), "got: {s}");
  }

  /// An accent over a label's base (`\!\(\*OverscriptBox[\(T\), \(^\)]\)`)
  /// is drawn as its combining mark on the base (`T̂`, `x̃`), not as a
  /// superscript beside it; a non-accent overscript stays a superscript.
  #[test]
  fn accent_overscript_text_label_is_a_combining_mark() {
    for (over, mark) in
      [("^", '\u{0302}'), ("~", '\u{0303}'), (".", '\u{0307}')]
    {
      let s = svg(&format!(
        "Graphics3D[{{Text[\"\\!\\(\\*OverscriptBox[\\(T\\), \\({over}\\)]\\)\", \
         {{0, 0, 0}}]}}]"
      ));
      assert!(s.contains(&format!(">T{mark}</text>")), "{over}: {s}");
      assert!(!s.contains("baseline-shift"), "{over}: {s}");
    }
    let s = svg(
      "Graphics3D[{Text[\"\\!\\(\\*OverscriptBox[\\(T\\), \\(k\\)]\\)\", \
       {0, 0, 0}]}]",
    );
    assert!(s.contains("baseline-shift=\"super\""), "got: {s}");
  }

  /// `BaseStyle` acts like a `Style` round the whole scene: a size (bare or
  /// as `FontSize -> n`) and a font face reach every `Text` label, also
  /// through `Show`, while a label's own size still wins.
  #[test]
  fn base_style_sizes_and_styles_text_labels() {
    for expr in [
      "Graphics3D[{Text[\"X\", {0, 0, 0}]}, BaseStyle -> 14]",
      "Graphics3D[{Text[\"X\", {0, 0, 0}]}, BaseStyle -> {14}]",
      "Graphics3D[{Text[\"X\", {0, 0, 0}]}, BaseStyle -> {FontSize -> 14}]",
      "Show[Graphics3D[{Text[\"X\", {0, 0, 0}]}], BaseStyle -> 14]",
    ] {
      let s = svg(expr);
      assert!(s.contains("font-size=\"14.0\""), "{expr}: {s}");
      assert!(!s.contains("font-size=\"12.0\""), "{expr}: {s}");
    }
    let s =
      svg("Graphics3D[{Style[Text[\"X\", {0, 0, 0}], 20]}, BaseStyle -> 14]");
    assert!(s.contains("font-size=\"20.0\""), "got: {s}");
    let s = svg("Graphics3D[{Text[\"X\", {0, 0, 0}]}, BaseStyle -> {Bold}]");
    assert!(s.contains("font-weight=\"bold\""), "got: {s}");
    let s = svg("Graphics3D[{Text[\"X\", {0, 0, 0}]}, BaseStyle -> {}]");
    assert!(s.contains("font-size=\"12.0\""), "got: {s}");
  }

  /// `Style[Text[…], FontSize -> n]` sizes the label like `Style[…, n]`.
  #[test]
  fn styled_text_label_font_size_rule() {
    let s = svg("Graphics3D[{Style[Text[\"X\", {0, 0, 0}], FontSize -> 18]}]");
    assert!(s.contains("font-size=\"18.0\""), "got: {s}");
  }

  #[test]
  fn arrow_tube_is_drawn() {
    let s = svg(
      "Graphics3D[{Orange, Arrow[Tube[{{0, 0, 0}, {1, 1, 0}, {1, 1, 1}}, \
       0.05]]}]",
    );
    assert!(polygon_count(&s) > 100);
  }

  /// A `Dynamic[x]` nested inside a primitive's coordinates
  /// (`Cuboid[{0.5 + Dynamic[3], …}, …]`, as `Dynamic[s] {i, j, k}`
  /// produces) or a directive's arguments (`Hue[Dynamic[h], 1, 1]`) is
  /// drawn at its current value — the same picture as the plain numbers,
  /// not a malformed cuboid that falls back to the unit cube.
  #[test]
  fn nested_dynamic_coordinates_and_colors_are_drawn_at_their_values() {
    let with_dynamic = svg(
      "Graphics3D[Table[{Hue[Dynamic[0.3] i, Dynamic[0.8], 1], \
       Cuboid[Dynamic[3] {i, 0, 0} - Dynamic[0.5]/2, \
       Dynamic[3] {i, 0, 0} + Dynamic[0.5]/2]}, {i, 3}], Boxed -> False]",
    );
    let plain = svg(
      "Graphics3D[Table[{Hue[0.3 i, 0.8, 1], \
       Cuboid[3 {i, 0, 0} - 0.5/2, 3 {i, 0, 0} + 0.5/2]}, {i, 3}], \
       Boxed -> False]",
    );
    assert_eq!(with_dynamic, plain);
    assert_eq!(polygon_count(&plain), polygon_count(&with_dynamic));
    assert!(fills(&with_dynamic).len() >= 3, "{with_dynamic}");
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

  /// A compiled function taking the point as one vector argument
  /// (`cf[{x, y, z}]`) is sampled through its compiled body, and draws the
  /// same surface as the expression it computes.
  #[test]
  fn compiled_vector_argument_body() {
    let plot = |body: &str| {
      clear_state();
      interpret(&format!(
        "cf = Compile[{{{{p, _Real, 1}}, {{s, _Real}}}}, s p.p]; \
         ExportString[ContourPlot3D[{body}, {{x, -1, 1}}, {{y, -1, 1}}, \
         {{z, -1, 1}}, Contours -> {{0.5}}, Mesh -> None, PlotPoints -> 12], \
         \"SVG\"]"
      ))
      .unwrap()
    };
    assert_eq!(plot("cf[{x, y, z}, 1.]"), plot("x^2 + y^2 + z^2"));
    // Entries may be constants or expressions in the variables.
    assert_eq!(
      plot("cf[{x, 2 y, 0.5}, 2.]"),
      plot("2. (x^2 + 4 y^2 + 0.25)")
    );
  }

  /// A tube thinner than the sampling grid (here a torus whose minor
  /// radius is well below a grid step) is traced whole: the grid catches
  /// it at only a few samples, so the cells around it are subdivided before
  /// the surface is traced. Regression: only those few samples produced
  /// triangles, so the ring came out as a dashed line of specks with wide
  /// gaps between them.
  #[test]
  fn tube_thinner_than_the_grid_is_traced_whole() {
    // The largest angular gap between the ring's vertices around its axis
    // and the largest distance of a vertex from the ring's center line.
    let ring = |plane: &str, center: &str, ranges: &str, points: u32| {
      clear_state();
      interpret(&format!(
        "pts = First[Cases[ContourPlot3D[\
           ({plane} - 1)^2 + ({center})^2 == 0.03^2, {ranges}, \
           Mesh -> None, PlotPoints -> {points}], \
           GraphicsComplex[p_, ___] :> p, Infinity]]; \
         {{Max[Differences[Sort[ArcTan @@@ pts[[All, {{1, 3}}]]]]], \
           Max[Abs[Sqrt[pts[[All, 1]]^2 + pts[[All, 3]]^2] - 1]]}}"
      ))
      .unwrap()
    };
    let check = |result: String| {
      let values: Vec<f64> = result
        .trim_matches(|c| c == '{' || c == '}')
        .split(", ")
        .map(|v| v.parse().unwrap())
        .collect();
      assert!(values[0] < 0.05, "the ring has a gap: {result}");
      assert!(values[1] < 0.05, "a vertex strays off the ring: {result}");
    };
    // A ring in the x-z plane, at a y that falls between grid planes.
    check(ring(
      "Sqrt[x^2 + z^2]",
      "y - 0.1",
      "{x, -1.2, 1.2}, {y, -0.7, 0.9}, {z, -1.2, 1.2}",
      30,
    ));
    // The same ring on a grid plane, sampled more coarsely across it.
    check(ring(
      "Sqrt[x^2 + z^2]",
      "y",
      "{x, -1.2, 1.2}, {y, -3, 1.2}, {z, -3, 1.2}",
      50,
    ));
  }

  /// `ColorFunction` colors each facet by `f[x, y, z, value]` with every
  /// argument scaled to 0..1, so a function of the scaled height paints
  /// the lower and upper half of a sphere differently. A named gradient
  /// colors by height too.
  #[test]
  fn color_function() {
    let sphere = "ContourPlot3D[x^2 + y^2 + z^2, {x, -1, 1}, {y, -1, 1}, \
                  {z, -1, 1}, Contours -> {0.8}, Mesh -> None";
    let reddish = |c: &String| {
      let v: Vec<u16> =
        c.split(',').map(|x| x.trim().parse().unwrap()).collect();
      v[0] > v[1] + 60 && v[0] > v[2] + 60
    };
    let bluish = |c: &String| {
      let v: Vec<u16> =
        c.split(',').map(|x| x.trim().parse().unwrap()).collect();
      v[2] > v[0] + 60 && v[2] > v[1] + 60
    };
    let halves = fills(&svg(&format!(
      "{sphere}, ColorFunction -> Function[{{x, y, z, f}}, \
       If[z > 0.5, Red, Blue]]]"
    )));
    assert!(halves.iter().any(reddish), "{halves:?}");
    assert!(halves.iter().any(bluish), "{halves:?}");
    // The value argument is passed too: one level gives one color.
    let by_value = fills(&svg(&format!(
      "{sphere}, ColorFunction -> Function[{{x, y, z, f}}, \
       If[f >= 0, Blue, Red]]]"
    )));
    assert!(by_value.iter().any(bluish) && !by_value.iter().any(reddish));
    let plain = fills(&svg(&format!("{sphere}]")));
    let rainbow =
      fills(&svg(&format!("{sphere}, ColorFunction -> \"Rainbow\"]")));
    assert!(!plain.iter().any(bluish));
    assert!(rainbow.iter().any(bluish) && rainbow.iter().any(reddish));
  }

  /// While the plot looks at its body with the variables still free, a
  /// compiled function given those variables in a vector stays
  /// unevaluated, instead of iterating its `FixedPoint` symbolically
  /// without end.
  #[test]
  fn compiled_vector_argument_iteration_is_not_run_symbolically() {
    clear_state();
    assert_eq!(
      interpret(
        "cf = Compile[{{p, _Real, 1}}, FixedPoint[(# + p.p/#)/2 &, 1., \
         SameTest -> (Abs[#1 - #2] < 10^-6 &)]]; \
         Head[ContourPlot3D[cf[{x, y, z}], {x, -1, 1}, {y, -1, 1}, \
         {z, -1, 1}, Contours -> {0.5}, PlotPoints -> 8]]"
      )
      .unwrap(),
      "Graphics3D"
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

  /// `AspectRatio -> Full` stretches the scene to the width and height
  /// `ImageSize` gives it, whatever the box's own proportions.
  #[test]
  fn aspect_ratio_full_fills_the_image_size() {
    let s = svg(
      "Graphics3D[{EdgeForm[Black], Cuboid[{-2, -2, 0}, {2, 2, 1}]}, \
       Boxed -> False, ViewPoint -> {0, -100, 0}, ImageSize -> {250, 350}, \
       AspectRatio -> Full, PlotRange -> {{-2, 2}, All, {0, 1}}]",
    );
    assert_eq!(size(&s), (250.0, 350.0));
    let ys: Vec<f64> = s
      .split(" y1=\"")
      .skip(1)
      .chain(s.split(" y2=\"").skip(1))
      .filter_map(|r| r.split('"').next()?.parse().ok())
      .collect();
    let lo = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // The unit-high slab fills nearly the whole height, not a quarter.
    assert!(hi - lo > 300.0, "{lo}..{hi}\n{s}");
  }

  /// The y range of the drawn `<polygon>` vertices in an SVG.
  fn polygon_y_extent(svg: &str) -> (f64, f64) {
    svg
      .split("<polygon points=\"")
      .skip(1)
      .filter_map(|rest| rest.split('"').next())
      .flat_map(|points| points.split_whitespace())
      .filter_map(|pair| pair.split(',').nth(1)?.parse::<f64>().ok())
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), y| {
        (lo.min(y), hi.max(y))
      })
  }

  /// Regression: an unboxed scene seen from an explicit `ViewPoint` at a
  /// fixed `ImageSize` was sphere-framed even with an explicit
  /// `AspectRatio`, so the projected box was not stretched to the picture
  /// and a squat box stayed a sliver.
  #[test]
  fn explicit_aspect_ratio_stretches_an_unboxed_aimed_scene() {
    let scene = "Graphics3D[Polygon[{{4, 0, 0}, {0, 4, 0}, {0, 0, 4}}], \
       Boxed -> False, ViewPoint -> {2, 1, .5}, BoxRatios -> {1, 1, 0.4}, \
       ImageSize -> {300, 300}";
    let (lo, hi) = polygon_y_extent(&svg(&format!("{scene}]")));
    let plain = hi - lo;
    let s = svg(&format!("{scene}, AspectRatio -> 1]"));
    assert_eq!(size(&s), (300.0, 300.0));
    let (lo, hi) = polygon_y_extent(&s);
    assert!(hi - lo > 1.8 * plain, "{plain} -> {}\n{s}", hi - lo);
  }

  /// Regression: `Plot3D` ignored an explicit `AspectRatio`: the picture
  /// took the requested shape, but the surface was not stretched to it.
  #[test]
  fn explicit_aspect_ratio_stretches_plot3d() {
    let plot = "Plot3D[4 - x - y, {x, 0, 4}, {y, 0, 4}, Mesh -> None, \
       Axes -> False, Boxed -> False, ImageSize -> 300";
    let plain = svg(&format!("{plot}]"));
    let (lo, hi) = polygon_y_extent(&plain);
    let plain_span = hi - lo;
    let plain_height = size(&plain).1;
    let s = svg(&format!("{plot}, AspectRatio -> 1]"));
    assert_eq!(size(&s), (300.0, 300.0));
    let (lo, hi) = polygon_y_extent(&s);
    // The surface grows with the picture's height.
    let expected = plain_span * 300.0 / plain_height;
    assert!(
      (hi - lo - expected).abs() < 0.05 * expected,
      "{} vs {expected}\n{s}",
      hi - lo
    );
  }

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

  // `Style[…, Bold, Italic]` on a 3D label reaches the drawn text, as it
  // does in 2D graphics (the slant and weight were dropped).
  #[test]
  fn styled_label_keeps_slant_and_weight() {
    for code in [
      "Graphics3D[{Inset[Text[Style[\"lbl\", 15, Italic, Bold, Red]], {0, 0, 0}]}]",
      "Graphics3D[{Text[Style[\"lbl\", Italic, Bold], {0, 0, 0}]}]",
    ] {
      let out = svg(code);
      let tag = out
        .split("<text ")
        .find(|t| t.contains(">lbl<"))
        .unwrap_or_else(|| panic!("no label: {out}"));
      assert!(tag.contains("font-style=\"italic\""), "{tag}");
      assert!(tag.contains("font-weight=\"bold\""), "{tag}");
    }
    let plain = svg("Graphics3D[{Text[\"lbl\", {0, 0, 0}]}]");
    assert!(!plain.contains("font-style=\"italic\""), "{plain}");
  }

  // A rational label is typeset as a stacked fraction in 3D, as it is in
  // 2D graphics: numerator and denominator on their own lines with a bar
  // between them, not `1/30` written on one line.
  #[test]
  fn rational_label_is_a_stacked_fraction() {
    for code in [
      "Graphics3D[{Text[1/30, {0, 0, 0}]}]",
      "Graphics3D[{Text[-7/4, {1, 2, 3}]}, Boxed -> False]",
    ] {
      let out = svg(code);
      assert!(!out.contains("/30<") && !out.contains("/4<"), "{out}");
      let texts: Vec<&str> = out
        .split("<text ")
        .skip(1)
        .map(|t| t.split('>').nth(1).unwrap_or("").split('<').next().unwrap())
        .collect();
      let (num, den) = if code.contains("1/30") {
        ("1", "30")
      } else {
        // A negative fraction sets its minus sign before the bar.
        assert!(texts.contains(&"\u{2212}"), "no minus: {texts:?}");
        ("7", "4")
      };
      assert!(texts.contains(&num) && texts.contains(&den), "{texts:?}");
      assert!(out.contains("<line "), "no fraction bar: {out}");
    }
  }

  // `Background -> color` paints a panel behind a 3D label, whether it is
  // the `Text`'s own option or a directive of a `Style` around the `Text`.
  #[test]
  fn label_background_paints_a_panel() {
    for code in [
      "Graphics3D[{Style[Text[1/2, {0, 0, 0}], 12, Background -> Yellow]}]",
      "Graphics3D[{Text[\"lbl\", {0, 0, 0}, Background -> Yellow]}]",
      "Graphics3D[{Text[Style[\"lbl\", 14], {0, 0, 0}, {0, 0}, \
       Background -> Yellow]}]",
    ] {
      let out = svg(code);
      let rect = out
        .find("fill=\"rgb(255,255,0)\"")
        .unwrap_or_else(|| panic!("no panel for {code}: {out}"));
      // The panel is drawn under the label, not over it.
      let text = out.rfind("<text ").unwrap();
      assert!(rect < text, "panel drawn over the label: {out}");
    }
    // The label's own option overrides the `Style` around it, and
    // `Background -> None` draws no panel.
    let own = svg(
      "Graphics3D[{Style[Text[\"a\", {0, 0, 0}, Background -> Red], \
       Background -> Yellow]}]",
    );
    assert!(own.contains("fill=\"rgb(255,0,0)\""), "{own}");
    assert!(!own.contains("fill=\"rgb(255,255,0)\""), "{own}");
    let none = svg(
      "Graphics3D[{Style[Text[\"a\", {0, 0, 0}, Background -> None], \
       Background -> Yellow]}]",
    );
    assert!(!none.contains("fill=\"rgb(255,255,0)\""), "{none}");
    let plain = svg("Graphics3D[{Text[\"a\", {0, 0, 0}]}]");
    assert!(!plain.contains("<rect x="), "{plain}");
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

  // Two faces crossing along a line through a corner of each are cut
  // there to be depth-sorted. The cut is not part of either outline, so no
  // dark line is drawn along it (here the vertical line x = 189.7).
  #[test]
  fn a_cut_through_a_corner_is_not_outlined() {
    let s = svg(
      "Graphics3D[{Polygon[{{-1, 0, -1}, {1, 0, -1}, {0, 0, 1}}], \
       Polygon[{{0, -1, -1}, {0, 1, -1}, {0, 0, 1}}]}, Boxed -> False]",
    );
    assert_eq!(polygon_count(&s), 4, "{s}");
    let outline: Vec<&str> =
      s.lines().filter(|l| l.starts_with("<line ")).collect();
    // Each piece keeps the two outline edges it got from its face.
    assert_eq!(outline.len(), 8, "{s}");
    assert!(
      s.lines()
        .filter(|l| l.starts_with("<polygon"))
        .all(|l| !l.contains("stroke=\"rgb(64,64,64)\"")),
      "{s}"
    );
    let x = |l: &str, attr: &str| {
      l.split(&format!("{attr}=\""))
        .nth(1)
        .and_then(|r| r.split('"').next())
        .map(str::to_string)
    };
    assert!(
      outline.iter().all(|l| x(l, "x1") != x(l, "x2")),
      "a line runs along the cut: {s}"
    );
  }

  // Faces that share an edge only up to the precision of their data (a
  // corner a millionth off the other face's plane) are not cut into
  // slivers against each other.
  #[test]
  fn faces_meeting_up_to_rounding_are_not_cut() {
    let s = svg(
      "Graphics3D[{Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}], \
       Polygon[{{0, 0, -1*^-6}, {1, 0, 1*^-6}, {1, 0.8, 0.3}, {0, 0.8, 0.3}}]}, \
       ViewPoint -> {0.3, 2, 3}, Boxed -> False]",
    );
    assert_eq!(polygon_count(&s), 4, "{s}");
  }

  // Rods that only cross on screen are not cut along each other's facet
  // planes: such cuts used up a facet's piece budget before the slab the
  // rod really passes through could cut it, so the rod showed through the
  // slab's side.
  #[test]
  fn crossing_rods_that_do_not_touch_are_not_cut() {
    let scene = |prims: &str| {
      polygon_count(&svg(&format!(
        "Graphics3D[{{EdgeForm[], {prims}}}, Boxed -> False, \
         ViewPoint -> {{-2.5, 2, 2}}]"
      )))
    };
    // Rod and bar each span the whole scene (23 units) on their own, so a
    // `Cylinder`'s scene-span ring cap cuts them the same in every scene.
    let slab = "Cuboid[{-6, -9, -4.5}, {6, 6, -3.5}]";
    let rod = "Cylinder[{{-4.5, -8, -12}, {-4.5, -8, 10}}, 0.5]";
    let bar = "Cylinder[{{-7.5, -11, 8.5}, {-7.5, 11, 8.5}}, 0.5]";
    let pierced = scene(&format!("{slab}, {rod}"));
    let alone = scene(bar);
    let all = scene(&format!("{slab}, {rod}, {bar}"));
    assert_eq!(all, pierced + alone);
  }

  // A rod through a slab is still cut by the slab's planes in a scene
  // with many other triangles: the split budget used to assume every
  // triangle could become the maximum number of pieces, keeping all but
  // the last few thousand triangles whole.
  #[test]
  fn a_pierced_slab_is_split_in_a_busy_scene() {
    let scene = |prims: &str| {
      polygon_count(&svg(&format!(
        "Graphics3D[{{EdgeForm[], {prims}}}, Boxed -> False, \
         ViewPoint -> {{-2.5, 2, 2}}]"
      )))
    };
    let pierced = "Cylinder[{{-4.5, -8, -12}, {-4.5, -8, 7}}, 0.5], \
                   Cuboid[{-6, -9, -4.5}, {6, 6, -3.5}]";
    let busy = "Table[Sphere[{40 + 3 i, 40, 0}, 1], {i, 24}]";
    let (a, b, both) = (
      scene(pierced),
      scene(busy),
      scene(&format!("{pierced}, {busy}")),
    );
    assert!(b > 6000, "{b}");
    // The far spheres change the scale, which can shift a few slivers;
    // an uncut rod would leave the scene hundreds of pieces short.
    assert!(both.abs_diff(a + b) <= 16, "{both} vs {a} + {b}");
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

  // The depth-sorting pieces of a translucent line are joined back into
  // one stroke: as separate elements, every joint shows an anti-aliasing
  // notch where the background bleeds through.
  #[test]
  fn translucent_line_is_one_stroke() {
    for line in [
      "Line[{{0, 0, 0}, {1, 1, 1}}]",
      "Line[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}]",
    ] {
      let s = svg(&format!(
        "Graphics3D[{{Opacity[0.5], Thickness[0.02], {line}}}, Boxed -> False]"
      ));
      assert_eq!(s.matches("<polyline ").count(), 1, "{line}: {s}");
      assert_eq!(s.matches("<line ").count(), 0, "{line}: {s}");
    }
    // Two separate lines stay two strokes.
    let s = svg(
      "Graphics3D[{Opacity[0.5], Line[{{0, 0, 0}, {1, 0, 0}}], \
       Line[{{0, 1, 1}, {1, 1, 1}}]}, Boxed -> False]",
    );
    assert_eq!(s.matches("<polyline ").count(), 2, "{s}");
  }

  // An opaque line is joined back into one stroke as well: drawn as many
  // pieces, the overlapping round caps of a finely sampled curve pile up
  // their anti-aliased edges until a thin line looks bold.
  #[test]
  fn opaque_line_is_one_stroke() {
    for line in [
      "Line[{{0, 0, 0}, {1, 1, 1}}]",
      "Line[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}]",
      "Line[Table[{Cos[t], Sin[t], t/10}, {t, 0, 2 Pi, Pi/100}]]",
    ] {
      for directives in ["", "Red, Thickness[0.01], "] {
        let s = svg(&format!(
          "Graphics3D[{{{directives}{line}}}, Boxed -> False]"
        ));
        assert_eq!(s.matches("<polyline ").count(), 1, "{line}: {s}");
        assert_eq!(s.matches("<line ").count(), 0, "{line}: {s}");
        assert!(s.contains("stroke-linecap=\"round\""), "{line}: {s}");
        assert!(!s.contains(" opacity="), "{line}: {s}");
      }
    }
  }

  // Segments that are already short on screen are not cut into further
  // depth-sorting pieces, so a densely sampled curve keeps about one
  // vertex per sample instead of eight.
  #[test]
  fn dense_line_is_not_subdivided() {
    let samples = 2001;
    let s = svg(&format!(
      "Graphics3D[Line[Table[{{Cos[t], Sin[t], t/10}}, \
       {{t, 0, 2 Pi, 2 Pi/{}}}]], Boxed -> False]",
      samples - 1
    ));
    let points = s
      .split("points=\"")
      .nth(1)
      .and_then(|p| p.split('"').next())
      .map_or(0, |p| p.split_whitespace().count());
    assert!(
      (samples..=samples + 64).contains(&points),
      "{points} vertices for {samples} samples"
    );
    // A long segment is still cut, so faces can be ordered against it.
    let s = svg(
      "Graphics3D[{Line[{{0, 0, 0}, {1, 1, 1}}], \
       Polygon[{{0, 1, 0}, {1, 0, 0}, {1, 1, 1}}]}, Boxed -> False]",
    );
    let points = s
      .split("<polyline points=\"")
      .skip(1)
      .map(|p| p.split('"').next().unwrap_or("").split_whitespace().count())
      .sum::<usize>();
    assert!(points > 2, "{s}");
  }

  // The triangles a translucent polygon is cut into are filled as one
  // path, so no light seam runs along the cut.
  #[test]
  fn translucent_polygon_is_one_fill() {
    for prims in [
      "EdgeForm[], Opacity[0.75], \
       Polygon[{{-1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, {1, -1, 0}}]",
      "Opacity[0.5], Red, \
       Polygon[{{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 2, 0}, {0, 2, 0}}]",
    ] {
      let s = svg(&format!("Graphics3D[{{{prims}}}, Boxed -> False]"));
      assert_eq!(s.matches("<path d=\"M").count(), 1, "{prims}: {s}");
      assert_eq!(s.matches("<polygon ").count(), 0, "{prims}: {s}");
    }
    // An opaque polygon keeps its own-color seam stroke instead.
    let s = svg(
      "Graphics3D[Polygon[{{-1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, {1, -1, 0}}], \
       Boxed -> False]",
    );
    assert!(!s.contains("<path d=\"M"), "{s}");
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

  // Regression: `Ticks -> False` drew the automatic ticks; like `None`,
  // it leaves every axis without tick labels.
  #[test]
  fn ticks_false_draws_no_tick_labels() {
    for ticks in ["False", "None", "{False, False, False}"] {
      let s = svg(&format!(
        "Plot3D[x + y, {{x, 0, 1}}, {{y, 0, 1}}, Ticks -> {ticks}]"
      ));
      assert!(!s.contains("<text"), "Ticks -> {ticks}: {s}");
    }
    let s = svg(
      "Plot3D[x + y, {x, 0, 1}, {y, 0, 1}, Ticks -> {False, Automatic, False}]",
    );
    assert!(s.contains(">0.5</text>"), "{s}");
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

  // A hidden edge that only meets a visible face at that face's corner
  // stays under the face: no stub of it is painted over the face.
  #[test]
  fn hidden_line_touching_a_face_corner_stays_hidden() {
    let s = svg(
      "Graphics3D[{Red, Line[{{0, 0, 0}, {0, 0, 1}}], White, Cuboid[]}, \
       ViewPoint -> {3, 3, 1}, Boxed -> False]",
    );
    let lines: Vec<&str> = s.lines().collect();
    let last_red = lines
      .iter()
      .rposition(|l| l.contains("stroke=\"rgb(255,0,0)\""))
      .expect("red line pieces");
    // The top face's two triangles start at its far corner, where the
    // hidden edge ends; both are painted over every piece of it.
    let top_face: Vec<usize> = lines
      .iter()
      .enumerate()
      .filter(|(_, l)| l.starts_with("<polygon points=\"180.0,6.0 "))
      .map(|(i, _)| i)
      .collect();
    assert_eq!(top_face.len(), 2, "{s}");
    assert!(top_face.iter().all(|&i| i > last_red), "{s}");
  }

  // A hidden edge that touches a neighboring visible face at its end is
  // not lifted over the face that hides the rest of it.
  #[test]
  fn hidden_line_touching_a_neighbor_face_stays_hidden() {
    let s = svg(
      "Graphics3D[{Red, Line[{{1, 0, 1}, {0, 0, 1}}], White, \
       Cuboid[{0, 0, 1}], Cuboid[{1, 0, 0}]}, ViewPoint -> {3, 3, 1}, \
       Boxed -> False]",
    );
    let lines: Vec<&str> = s.lines().collect();
    let last_red = lines
      .iter()
      .rposition(|l| l.contains("stroke=\"rgb(255,0,0)\""))
      .expect("red line pieces");
    // The upper cube's front-left face, which the edge runs behind.
    let hiding_face = lines
      .iter()
      .position(|l| {
        // Its fill may reach across the internal cut by extra corners.
        l.starts_with("<polygon points=\"134.0,170.5 ")
          && l.contains(" 247.1,32.4 245.5,196.0\"")
      })
      .expect("hiding face");
    assert!(last_red < hiding_face, "{s}");
  }

  // The fan triangles of a translucent cap touch the outline on one side
  // only. Stroking them in their own color overlapped the neighbors along
  // the internal cuts and drew dark spokes across the cap; the outline is
  // stroked on its own.
  #[test]
  fn translucent_triangles_on_the_outline_have_no_seam_stroke() {
    for solid in [
      "Cylinder[]",
      "Cone[]",
      "Cuboid[]",
      "Tube[{{0, 0, 0}, {1, 1, 1}}, 0.2]",
    ] {
      let s = svg(&format!(
        "Graphics3D[{{Opacity[0.5], Red, {solid}}}, Boxed -> False]"
      ));
      let seamed: Vec<&str> = s
        .lines()
        .filter(|l| l.starts_with("<polygon") && l.contains(" opacity=\""))
        .filter(|l| !l.contains("stroke=\"none\""))
        .collect();
      assert!(seamed.is_empty(), "{solid}: {:?}", seamed.first());
      assert!(s.contains(" opacity=\"0.5\""), "{solid}: {s}");
    }
  }

  /// The `stroke-width` values of the black `<line>` pieces of an SVG.
  fn black_line_widths(svg: &str) -> Vec<f64> {
    svg
      .lines()
      .filter(|l| {
        (l.starts_with("<line") || l.starts_with("<polyline"))
          && l.contains("stroke=\"rgb(0,0,0)\"")
      })
      .filter_map(|l| {
        l.split("stroke-width=\"")
          .nth(1)
          .and_then(|s| s.split('"').next())
          .and_then(|w| w.parse::<f64>().ok())
      })
      .collect()
  }

  // A 3D line with no thickness directive is a hairline, while an explicit
  // thickness still sets its width.
  #[test]
  fn default_line_is_a_hairline() {
    let widths = black_line_widths(&svg(
      "Graphics3D[{Black, Line[{{0, 0, 0}, {1, 1, 1}}]}, Boxed -> False]",
    ));
    assert!(
      !widths.is_empty() && widths.iter().all(|w| *w == 0.5),
      "{widths:?}"
    );
    let widths = black_line_widths(&svg(
      "Graphics3D[{Black, AbsoluteThickness[3], \
       Line[{{0, 0, 0}, {1, 1, 1}}]}, Boxed -> False]",
    ));
    assert!(
      !widths.is_empty() && widths.iter().all(|w| *w == 3.0),
      "{widths:?}"
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

  /// The vertical extent of a `Plot3D` from `AbsoluteOptions`.
  fn z_range(code: &str) -> (f64, f64) {
    let out = interpret(&format!(
      "Last[PlotRange /. AbsoluteOptions[{code}, PlotRange]]"
    ))
    .unwrap();
    let inner = out.trim_matches(|c| c == '{' || c == '}');
    let mut parts = inner.split(',').map(|p| {
      let p = p.trim();
      match p.split_once("*^") {
        Some((m, e)) => {
          m.parse::<f64>().unwrap() * 10f64.powi(e.parse::<i32>().unwrap())
        }
        None => p.parse::<f64>().unwrap(),
      }
    });
    (parts.next().unwrap(), parts.next().unwrap())
  }

  // The automatic vertical range clips outlying heights off, as WL's
  // does: a pole or a jump to a far value along one edge of the domain
  // leaves the box spanning the bulk of the surface.
  #[test]
  fn automatic_z_range_clips_outliers() {
    let (lo, hi) = z_range("Plot3D[1/(x^2 + y^2), {x, -1, 1}, {y, -1, 1}]");
    assert_eq!(lo, 0.5);
    assert!(hi > 5.0 && hi < 20.0, "{hi}");
    // A cliff along the far edge (y > 0.98) is dropped from the range.
    let (lo, hi) =
      z_range("Plot3D[If[y > 0.98, -10, x + y], {x, 0, 1}, {y, 0, 1}]");
    assert!(lo > -2.0 && lo < 0.0, "{lo}");
    assert!((hi - 1.98).abs() < 1e-9, "{hi}");
  }

  // `PlotRange -> All`, also in the vertical slot of a three-axis spec,
  // keeps every sampled height; a smooth surface keeps its whole extent.
  #[test]
  fn z_range_all_keeps_outliers() {
    for opt in ["PlotRange -> All", "PlotRange -> {Full, Full, All}"] {
      let (lo, hi) = z_range(&format!(
        "Plot3D[1/(x^2 + y^2), {{x, -1, 1}}, {{y, -1, 1}}, {opt}]"
      ));
      assert_eq!(lo, 0.5);
      assert!(hi > 600.0, "{opt}: {hi}");
    }
    let (lo, hi) = z_range("Plot3D[x y, {x, -1, 1}, {y, -1, 1}]");
    assert_eq!((lo, hi), (-1.0, 1.0));
  }

  // Regression: the `Graphics3D` a `Plot3D` returns kept the whole sampled
  // surface though a `RegionFunction` had cut it, so `Show` of the plot drew
  // the excluded part too. Its points now stay in the region, and the cells
  // along the region's edge are clipped to it as in the standalone render.
  #[test]
  fn region_function_cuts_the_returned_surface() {
    clear_state();
    let plot = "p = Plot3D[4 - x - y, {x, 0, 4}, {y, 0, 4}, \
       PlotRange -> {0, 4}, RegionFunction -> Function[{x, y, z}, z > 0], \
       Mesh -> None]; pts = Flatten[Cases[p, GraphicsComplex[v_, ___] :> v, \
       Infinity], 1];";
    assert_eq!(
      interpret(&format!(
        "{plot} {{Max[pts[[All, 1]] + pts[[All, 2]]] <= 4 + 10^-9, \
         Max[pts[[All, 1]] + pts[[All, 2]]] > 3.99, \
         Min[pts[[All, 1]]] == 0., Min[pts[[All, 2]]] == 0.}}"
      ))
      .unwrap(),
      "{True, True, True, True}"
    );
    // Shown alone, the plot keeps the same cut.
    assert_eq!(
      interpret(
        "pts2 = Flatten[Cases[Show[p], GraphicsComplex[v_, ___] :> v, \
         Infinity], 1]; Max[pts2[[All, 1]] + pts2[[All, 2]]] <= 4 + 10^-9"
      )
      .unwrap(),
      "True"
    );
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

  // A numeric filling level outside the surface's own values widens the
  // automatic vertical range so the wall reaches it.
  #[test]
  fn filling_level_below_the_surface_extends_the_range() {
    clear_state();
    let out = interpret(
      "Cases[Plot3D[x^2, {x, -1, 1}, {y, -1, 1}, Mesh -> None, \
       Filling -> -3], Polygon[q_?(ArrayDepth[#] == 3 &)] :> \
       Min[q[[All, All, 3]]], Infinity]",
    )
    .unwrap();
    assert_eq!(out, "{-3.}");
  }

  // `FillingStyle` colors the wall, which travels into `Show` with the
  // surface as polygons in data coordinates.
  #[test]
  fn filling_style_colors_the_wall_and_survives_show() {
    clear_state();
    let out = interpret(
      "Cases[Plot3D[y, {x, 0, 1}, {y, 0, 1}, Filling -> Bottom, \
       FillingStyle -> Red], {EdgeForm[], c_RGBColor, Polygon[_]} :> c, \
       Infinity]",
    )
    .unwrap();
    assert_eq!(out, "{RGBColor[1., 0., 0.]}");
    let red = svg(
      "Plot3D[y, {x, 0, 1}, {y, 0, 1}, Filling -> Bottom, \
       FillingStyle -> Red]",
    );
    assert!(!red.contains("fill=\"rgb(178,178,178)\""));
    // The lit wall is red-dominated.
    assert!(fills(&red).iter().any(|c| {
      let v: Vec<u32> = c.split(',').filter_map(|x| x.parse().ok()).collect();
      v.len() == 3 && v[0] > 150 && v[1] < 100 && v[2] < 100
    }));
    let shown = svg(
      "Show[Plot3D[y, {x, 0, 1}, {y, 0, 1}, Filling -> Bottom], \
       Graphics3D[Sphere[{0, 0, 0}, 0.1]]]",
    );
    assert!(
      polygon_count(&shown)
        > polygon_count(&svg(
          "Show[Plot3D[y, {x, 0, 1}, {y, 0, 1}], \
       Graphics3D[Sphere[{0, 0, 0}, 0.1]]]"
        ))
    );
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

  // A `Text` inside a `GraphicsComplex` is anchored at the indexed point.
  #[test]
  fn graphics_complex_text_by_index() {
    let s = svg(
      "Graphics3D[GraphicsComplex[{{0, 0, 0}, {1, 1, 1}}, \
       {Sphere[1, 0.1], Text[\"Qz\", 2]}]]",
    );
    assert!(s.contains(">Qz<"), "{s}");
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

  // `Opacity` fades a `Text` label in a 3-D scene; `Opacity[0]` hides it.
  #[test]
  fn opacity_applies_to_text_labels() {
    let shown = svg("Graphics3D[{Text[\"Qz\", {0, 0, 0}], Sphere[]}]");
    assert!(shown.contains(">Qz<"));
    let hidden =
      svg("Graphics3D[{Opacity[0], Text[\"Qz\", {0, 0, 0}], Sphere[]}]");
    assert!(!hidden.contains(">Qz<"));
    let faded = svg("Graphics3D[{Opacity[0.3], Text[\"Qz\", {0, 0, 0}]}]");
    assert!(faded.contains("opacity=\"0.3\">Qz<"), "{faded}");
  }

  /// The area of the first filled arrowhead `<polygon>` (the outline after
  /// a `<polyline>`), by the shoelace formula.
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
    let n = p.len();
    (0..n)
      .map(|i| {
        let (a, b) = (p[i], p[(i + 1) % n]);
        a.0 * b.1 - b.0 * a.1
      })
      .sum::<f64>()
      .abs()
      / 2.0
  }

  // `Arrowheads[s]` sizes the heads of 3D arrows as a fraction of the
  // diagonal of the scene's box; the default is a twentieth.
  #[test]
  fn arrowheads_size_3d_arrow_heads() {
    let default = svg("Graphics3D[{Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    let same =
      svg("Graphics3D[{Arrowheads[0.05], Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    let large =
      svg("Graphics3D[{Arrowheads[0.15], Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    assert!((head_area(&default) - head_area(&same)).abs() < 1.0);
    assert!(head_area(&large) > 6.0 * head_area(&default));
  }

  /// The first point of every arrowhead `<polygon>` after the arrow's
  /// `<polyline>` (each head's outline starts at its tip).
  fn head_tips(svg: &str) -> Vec<(f64, f64)> {
    let after = svg.split("<polyline").nth(1).unwrap();
    after
      .split("<polygon points=\"")
      .skip(1)
      .map(|pts| {
        let first = pts.split(' ').next().unwrap();
        let mut it = first.split(',').map(|v| v.parse::<f64>().unwrap());
        (it.next().unwrap(), it.next().unwrap())
      })
      .collect()
  }

  // `Arrowheads[{s1, s2, …}]` spreads its heads evenly from tail to tip, and
  // a negative size turns its head back along the arrow, so `{-s, s}` is a
  // double-headed 3D arrow sized like `Arrowheads[s]` (rather than one
  // default-size head at the tip).
  #[test]
  fn arrowheads_list_puts_heads_along_3d_arrow() {
    let arrow = |heads: &str| {
      svg(&format!(
        "Graphics3D[{{Arrowheads[{heads}], Arrow[{{{{0, 0, 0}}, {{0, 0, 1}}}}], \
         Point[{{{{-1, -1, -1}}, {{1, 1, 1}}}}]}}, Boxed -> False]"
      ))
    };
    let single = arrow("0.02");
    let double = arrow("{-0.02, 0.02}");
    assert_eq!(head_tips(&single).len(), 1);
    let tips = head_tips(&double);
    assert_eq!(tips.len(), 2, "{double}");
    // The heads keep the size the list gives them, not the default
    // (a twentieth).
    let default = head_area(&arrow("0.05"));
    assert!(head_area(&double) < 0.5 * default, "{double}");
    // One tip sits at the arrow's head end, the other at its tail end (the
    // z axis points up the picture, so the tail is lower on screen).
    let single_tip = head_tips(&single)[0];
    assert!(
      tips.iter().any(|t| (t.0 - single_tip.0).abs() < 0.2
        && (t.1 - single_tip.1).abs() < 0.2)
    );
    assert!(tips.iter().any(|t| t.1 > single_tip.1 + 20.0), "{tips:?}");
    // Three entries put the middle head halfway along.
    assert_eq!(head_tips(&arrow("{0.02, 0.02, 0.02}")).len(), 3);
  }

  // The head of a 3D line arrow is a cone sized by the scene, not a fixed
  // number of pixels, so it stays about as large on screen however big the
  // scene is.
  #[test]
  fn arrow3d_head_scales_with_the_scene() {
    let small = svg("Graphics3D[{Arrow[{{0, 0, 0}, {1, 0, 0}}]}]");
    let big = svg(
      "Graphics3D[{Arrow[{{0, 0, 0}, {1, 0, 0}}], \
       Point[{{-10, -10, -10}, {10, 10, 10}}]}]",
    );
    // A twentieth of the 1-unit scene is well over the old fixed 8-pixel
    // head (an area of 24).
    assert!(head_area(&small) > 60.0, "{}", head_area(&small));
    // In the 35-times larger scene the 1-unit arrow shrinks on screen but
    // its head (a twentieth of the scene) stays about as large.
    assert!(head_area(&big) > 0.3 * head_area(&small));
  }

  // A head keeps its size on an arrow shorter than the head itself (up to
  // the perspective of its slightly different place in the scene).
  #[test]
  fn arrow3d_head_not_clamped_to_short_arrow() {
    let scene = |tip: &str| {
      svg(&format!(
        "Graphics3D[{{Arrow[{{{{0, 0, 0}}, {tip}}}], \
         Point[{{{{-1, -1, -1}}, {{1, 1, 1}}}}]}}]"
      ))
    };
    let short = head_area(&scene("{0.01, 0, 0}"));
    let long = head_area(&scene("{0.9, 0, 0}"));
    assert!((short - long).abs() < 0.2 * long, "{short} vs {long}");
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

  // A parametric surface keeps its smooth shading when `Show` combines it
  // with other graphics (merging its `GraphicsComplex` of grid quads)
  // instead of drawing the plot's own picture.
  #[test]
  fn parametric_surface_in_show_is_smooth() {
    let torus = "ParametricPlot3D[{Cos[u] (3 + Cos[v]), Sin[v], \
                 Sin[u] (3 + Cos[v])}, {u, 0, 2 Pi}, {v, 0, 2 Pi}, \
                 Mesh -> False]";
    let s = svg(&format!(
      "Show[{torus}, Graphics3D[Sphere[{{0, 0, 0}}, 0.5]]]"
    ));
    let gradients = s.matches("fill=\"url(#").count();
    assert!(
      gradients * 2 > polygon_count(&s),
      "{gradients} of {}",
      polygon_count(&s)
    );
  }

  // Shared-vertex quads in a `GraphicsComplex` without outlines are shaded
  // as one smooth surface; the hard fold of a box keeps flat faces.
  #[test]
  fn graphics_complex_quads_are_smooth() {
    let n = 12;
    let pts = (0..=n)
      .flat_map(|i| {
        (0..=1).map(move |j| {
          let a = std::f64::consts::PI * i as f64 / n as f64;
          format!("{{{:.6}, {:.6}, {j}}}", a.cos(), a.sin())
        })
      })
      .collect::<Vec<_>>()
      .join(", ");
    let quads = (0..n)
      .map(|i| {
        let k = 2 * i + 1;
        format!("{{{}, {}, {}, {}}}", k, k + 2, k + 3, k + 1)
      })
      .collect::<Vec<_>>()
      .join(", ");
    let s = svg(&format!(
      "Graphics3D[GraphicsComplex[{{{pts}}}, {{EdgeForm[], \
       Polygon[{{{quads}}}]}}]]"
    ));
    assert!(s.contains("<linearGradient"), "{s}");
    let cube = svg(
      "Graphics3D[GraphicsComplex[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, \
       {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}, \
       {EdgeForm[], Polygon[{{1, 2, 3, 4}, {5, 6, 7, 8}, {1, 2, 6, 5}, \
       {2, 3, 7, 6}, {3, 4, 8, 7}, {4, 1, 5, 8}}]}]]",
    );
    assert!(!cube.contains("<linearGradient"), "{cube}");
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

  // A translucent face lying on an opaque one stays beneath it, even when
  // it is given later: only the opaque face's color shows through.
  #[test]
  fn translucent_face_on_opaque_face_stays_beneath() {
    for (first, second) in [
      ("Red, ", "Blue, Opacity[0.5], "),
      ("Blue, Opacity[0.5], ", "Opacity[1], Red, "),
    ] {
      let s = svg(&format!(
        "Graphics3D[{{{first}Polygon[{{{{0, 0, 0}}, {{2, 0, 0}}, \
         {{2, 2, 0}}, {{0, 2, 0}}}}], {second}Polygon[{{{{0, 0, 0}}, \
         {{2, 0, 0}}, {{2, 2, 0}}, {{0, 2, 0}}}}]}}, Boxed -> False]"
      ));
      let polys: Vec<&str> = s.split("<polygon ").skip(1).collect();
      assert!(!polys.is_empty(), "{s}");
      assert!(
        !polys.last().unwrap().contains("fill-opacity"),
        "the opaque face must be drawn last: {s}"
      );
    }
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

// A dense opaque volume is drawn at full resolution: averaging blocks of
// ±1 voxels (clamped to black and white) would turn them gray.
#[test]
fn dense_raster_volume_keeps_every_voxel_color() {
  let s = svg(
    "Graphics3D[Raster3D[Table[(-1)^(i + j + k), {i, 16}, {j, 16}, {k, 16}]]]",
  );
  let fills: std::collections::BTreeSet<&str> = s
    .split("<polygon")
    .skip(1)
    .filter_map(|p| p.split("fill=\"").nth(1)?.split('"').next())
    .collect();
  assert_eq!(
    fills.into_iter().collect::<Vec<_>>(),
    ["rgb(0,0,0)", "rgb(255,255,255)"]
  );
}

// Equal neighbors on a face of a solid volume merge, so a uniform volume is
// just its six box faces however many voxels it has.
#[test]
fn uniform_raster_volume_is_six_faces() {
  let s = svg("Graphics3D[Raster3D[ConstantArray[0.5, {20, 20, 20}]]]");
  assert_eq!(s.matches("<polygon").count(), 12);
  assert!(s.contains("fill=\"rgb(128,128,128)\""));
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

mod internal_cut_seams {
  use super::*;

  /// The point count of every filled `<polygon>` an SVG draws.
  fn polygon_point_counts(svg: &str) -> Vec<usize> {
    svg
      .split("<polygon points=\"")
      .skip(1)
      .filter_map(|s| s.split('"').next())
      .map(|pts| pts.split_whitespace().count())
      .collect()
  }

  const SQUARE: &str = "Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}]";

  /// Each half of a fan-triangulated face reaches across the diagonal it
  /// was split along with a thin band, so the halves overlap and no
  /// anti-aliasing seam traces the cut.
  #[test]
  fn fan_halves_overlap_across_the_cut() {
    let s = svg(&format!("Graphics3D[{SQUARE}, Boxed -> False]"));
    assert_eq!(polygon_point_counts(&s), vec![5, 5], "{s}");
  }

  /// Every triangle of a hexagon fan widens across each of its cuts: the
  /// end triangles have one, the middle ones two.
  #[test]
  fn every_cut_of_a_larger_fan_widens() {
    let s = svg(
      "Graphics3D[Polygon[Table[{Cos[t], Sin[t], 0}, {t, 0, 5 Pi/3, Pi/3}]], Boxed -> False]",
    );
    let mut counts = polygon_point_counts(&s);
    counts.sort_unstable();
    assert_eq!(counts, vec![5, 5, 7, 7], "{s}");
  }

  /// Cube faces are fan-triangulated quads as well.
  #[test]
  fn cuboid_faces_widen() {
    let s = svg("Graphics3D[Cuboid[], Boxed -> False]");
    let counts = polygon_point_counts(&s);
    assert_ne!(counts, [] as [usize; 0]);
    assert!(counts.iter().all(|&n| n == 5), "{counts:?}");
  }

  /// A lone triangle has no internal cut to widen.
  #[test]
  fn lone_triangle_keeps_three_corners() {
    let s = svg(
      "Graphics3D[Polygon[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}], Boxed -> False]",
    );
    assert_eq!(polygon_point_counts(&s), vec![3], "{s}");
  }

  /// Overlapping translucent halves would darken the diagonal, and without
  /// a drawn outline the band could show past the face, so neither widens.
  #[test]
  fn translucent_or_edgeless_faces_keep_three_corners() {
    for style in ["Opacity[0.5]", "EdgeForm[]"] {
      let s = svg(&format!(
        "Graphics3D[{{{style}, {SQUARE}}}, Boxed -> False]"
      ));
      // A translucent face's pieces are filled as one `<path>` instead.
      let counts = polygon_point_counts(&s);
      assert!(counts.iter().all(|&n| n == 3), "{style}: {s}");
      if style == "EdgeForm[]" {
        assert_eq!(counts, vec![3, 3], "{style}: {s}");
      }
    }
  }
}

/// A "perforated" face is one polygon running round its outline, across a
/// bridge to an inner copy shrunk toward the center, and round that copy
/// (`Join[pts, {p1}, inner, {q1}]`). Wolfram fills it with the even-odd
/// rule, which also settles the two extremes of the shrink factor.
mod perforated_faces {
  use super::*;

  /// The corners as a Wolfram list `{{x, y, z}, …}`.
  fn coords(outline: &[[i64; 3]]) -> String {
    let pts: Vec<String> = outline
      .iter()
      .map(|p| format!("{{{}, {}, {}}}", p[0], p[1], p[2]))
      .collect();
    format!("{{{}}}", pts.join(", "))
  }

  /// The face's corners, then the inner ring `{1 - d, d} . {p, center}`.
  fn perforated(outline: &[[i64; 3]], d: &str) -> String {
    let pts = coords(outline);
    format!(
      "With[{{pts = {pts}, d = {d}}}, Module[{{inner = Map[{{1 - d, d}} . {{#, Mean[pts]}} &, pts]}}, \
       Polygon[Join[pts, pts[[{{1}}]], inner, inner[[{{1}}]]]]]]"
    )
  }

  const SQUARE: [[i64; 3]; 4] = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]];
  const TRIANGLE: [[i64; 3]; 3] = [[0, 0, 0], [3, 0, 1], [0, 3, 2]];

  fn scene(prims: &str) -> String {
    svg(&format!("Graphics3D[{{Yellow, {prims}}}, Boxed -> False]"))
  }

  /// With the inner ring collapsed onto the center (`d = 1`) the hole
  /// encloses nothing: the face is drawn exactly as the plain polygon,
  /// without the bridge stroked from a corner to the center.
  #[test]
  fn collapsed_hole_draws_the_plain_face() {
    for outline in [&SQUARE[..], &TRIANGLE[..]] {
      let plain = format!("Polygon[{}]", coords(outline));
      for d in ["1", "1.0"] {
        assert_eq!(
          scene(&perforated(outline, d)),
          scene(&plain),
          "d = {d} for {plain}"
        );
      }
    }
  }

  /// With the inner ring on the outline itself (`d = 0`) the face is
  /// enclosed twice, so the even-odd rule leaves it unfilled and only its
  /// outline is drawn — as `FaceForm[None]` draws it.
  #[test]
  fn retraced_outline_leaves_the_face_empty() {
    for outline in [&SQUARE[..], &TRIANGLE[..]] {
      let hollow = format!("FaceForm[None], Polygon[{}]", coords(outline));
      for d in ["0", "0.0"] {
        let s = scene(&perforated(outline, d));
        assert_eq!(s, scene(&hollow), "d = {d} for {hollow}");
        // Any polygon left is an unfilled outline.
        assert!(
          s.lines()
            .filter(|l| l.starts_with("<polygon"))
            .all(|l| l.contains("fill-opacity=\"0\"")),
          "face still filled: {s}"
        );
        assert!(
          s.contains("stroke=\"rgb(64,64,64)\""),
          "outline missing: {s}"
        );
      }
    }
  }

  /// In between, the inner ring is a hole: the face stays filled but
  /// takes more pieces than the solid face to go round it.
  #[test]
  fn partial_shrink_cuts_a_hole() {
    let holed = scene(&perforated(&SQUARE, "0.5"));
    let solid = scene(&perforated(&SQUARE, "1"));
    assert!(!holed.contains("fill-opacity=\"0\""));
    assert!(polygon_count(&holed) > polygon_count(&solid));
  }
}
