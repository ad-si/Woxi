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

  /// `Axes -> Automatic` draws the three labeled axes of a 3D scene, the
  /// same as `Axes -> True` (Demonstrations often pass it to `Show`).
  /// Regression: it was ignored, leaving the box without ticks or labels.
  #[test]
  fn axes_automatic_draws_the_axes() {
    let with = |axes: &str| {
      svg(&format!(
        "Graphics3D[Sphere[], Axes -> {axes}, AxesLabel -> {{\"x\", \"y\", \"z\"}}]"
      ))
    };
    let automatic = with("Automatic");
    assert!(automatic.matches("<text").count() > 3);
    assert_eq!(automatic, with("True"));
    assert_eq!(with("False").matches("<text").count(), 0);
  }

  /// `Axes -> None` hides the axes of a 3D scene exactly like
  /// `Axes -> False`. Regression: only `False` was recognized, so the 3D
  /// plot functions (whose axes default to on) kept drawing them, and
  /// `ListPointPlot3D` / `ListLinePlot3D` ignored the option altogether.
  #[test]
  fn axes_none_hides_the_axes() {
    for plot in [
      "Graphics3D[Sphere[], Axes -> {}]",
      "Plot3D[x y, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 5, Axes -> {}]",
      "ParametricPlot3D[{Cos[u] (2 + Cos[v]), Sin[u] (2 + Cos[v]), Sin[v]}, \
        {u, 0, 2 Pi}, {v, 0, 2 Pi}, PlotPoints -> 8, Axes -> {}]",
      "ParametricPlot3D[{Cos[u], Sin[u], u}, {u, 0, 6}, Axes -> {}]",
      "ListPlot3D[{{1, 2}, {3, 4}}, Axes -> {}]",
      "ListPointPlot3D[{{1, 2, 3}, {3, 4, 1}, {2, 2, 2}}, Axes -> {}]",
      "ListLinePlot3D[{{1, 2, 3}, {2, 3, 1}, {3, 1, 2}}, Axes -> {}]",
      "ContourPlot3D[x^2 + y^2 + z^2 == 1, {x, -1, 1}, {y, -1, 1}, \
        {z, -1, 1}, PlotPoints -> 6, Axes -> {}]",
      "RevolutionPlot3D[t^2, {t, 0, 1}, PlotPoints -> 6, Axes -> {}]",
    ] {
      let with = |axes: &str| svg(&plot.replace("{}", axes));
      let none = with("None");
      assert_eq!(none.matches("<text").count(), 0, "{plot}");
      assert_eq!(none, with("False"), "{plot}");
      assert!(with("True").matches("<text").count() > 3, "{plot}");
    }
  }

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

  /// A see-through sphere is tessellated about the line of sight like an
  /// opaque one, so its near and far halves meet on the outline: seen
  /// head-on, its facets' corners lie on a few rings around the center,
  /// the outermost being the finely divided equator. Regression: only
  /// opaque spheres were, so a translucent one was cut along a slanted,
  /// saw-toothed path through its latitude bands, whose facets, lit half
  /// from the front and half from behind, showed as a jagged rim.
  #[test]
  fn translucent_sphere_halves_meet_on_its_outline() {
    for view in ["{1.3, -2.4, 2}", "{0, 0, 10}", "{3, -1, 0.5}"] {
      for opacity in ["0.2", "0.6"] {
        let s = svg(&format!(
          "Graphics3D[{{Opacity[{opacity}], Sphere[]}}, Boxed -> False, \
           ViewPoint -> {view}]"
        ));
        let pts: Vec<(f64, f64)> = s
          .split("<polygon points=\"")
          .skip(1)
          .filter_map(|p| p.split('"').next())
          .flat_map(|p| p.split_whitespace())
          .filter_map(|p| {
            let (x, y) = p.split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
          })
          .collect();
        let span = |f: fn(&(f64, f64)) -> f64| {
          let v = pts.iter().map(f);
          (
            v.clone().fold(f64::MAX, f64::min),
            v.fold(f64::MIN, f64::max),
          )
        };
        let ((x0, x1), (y0, y1)) = (span(|p| p.0), span(|p| p.1));
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        let radius = (x1 - x0).max(y1 - y0) / 2.0;
        let dist =
          |p: &(f64, f64)| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt();
        // Corner distances from the center, in hundredths of the radius.
        let mut rings: Vec<i64> = pts
          .iter()
          .map(|p| (100.0 * dist(p) / radius).round() as i64)
          .collect();
        rings.sort_unstable();
        rings.dedup();
        // A slanted latitude/longitude mesh spreads its corners over
        // nearly every distance.
        assert!(
          rings.len() < 40,
          "{view}, opacity {opacity}: corners on {} rings",
          rings.len()
        );
        // The outline is the equator, not a coarse cut.
        let mut rim: Vec<(i64, i64)> = pts
          .iter()
          .filter(|p| dist(p) > 0.985 * radius)
          .map(|p| ((p.0 * 10.0).round() as i64, (p.1 * 10.0).round() as i64))
          .collect();
        rim.sort_unstable();
        rim.dedup();
        assert!(
          rim.len() >= 48,
          "{view}, opacity {opacity}: {} corners on the outline",
          rim.len()
        );
      }
    }
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
  /// arrow in a large scene keeps a visible head. The head is drawn
  /// without a rim round its base.
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
         EdgeForm[], Cone[{{{{{b}, 0, 0}}, {{1, 0, 0}}}}, {r}]}}, {range}]",
        b = 1.0 - d,
        r = 0.22 * d
      ))
    );
  }

  /// A `Tube` without a radius is as thick as a fixed fraction (0.0035) of
  /// the diagonal of the scene's box, not of its own curve.
  #[test]
  fn tube_default_radius_is_sized_by_the_scene() {
    let r = 0.0035 * 3f64.sqrt() * 4.0;
    let range = "PlotRange -> {{-2, 2}, {-2, 2}, {-2, 2}}";
    assert_eq!(
      svg(&format!(
        "Graphics3D[Tube[{{{{0, 0, 0}}, {{1, 0, 0}}}}], {range}]"
      )),
      svg(&format!(
        "Graphics3D[Tube[{{{{0, 0, 0}}, {{1, 0, 0}}}}, {r}], {range}]"
      ))
    );
  }

  /// `Tube[pts, r, VertexColors -> {c1, c2}]` blends the colors along the
  /// tube in twelve steps per segment, each the mean of the colors at its
  /// two ends. Regression: the option was ignored, so the gradient bonds of
  /// a crystal-structure Demonstration came out in the plain (black) color.
  #[test]
  fn tube_vertex_colors_blend_along_the_tube() {
    let s = svg(
      "Graphics3D[{Black, Tube[{{0, 0, 0}, {1, 0, 0}}, 0.1, \
       VertexColors -> {Red, Blue}]}, Lighting -> {{\"Ambient\", White}}]",
    );
    let mut expected: std::collections::BTreeSet<String> = (0..12)
      .map(|k| {
        let f = f64::from(2 * k + 1) / 24.0;
        let (r, b) = ((255.0 * (1.0 - f)).round(), (255.0 * f).round());
        format!("{r},0,{b}")
      })
      .collect();
    expected.insert("255,255,255".to_string());
    assert_eq!(fills(&s), expected);
  }

  /// Every segment of a longer tube blends between the colors of its own
  /// two points.
  #[test]
  fn tube_vertex_colors_blend_per_segment() {
    let s = svg(
      "Graphics3D[{Tube[{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, 0.1, \
       VertexColors -> {Red, Green, Blue}]}, \
       Lighting -> {{\"Ambient\", White}}]",
    );
    let colors = fills(&s);
    assert_eq!(colors.len(), 25, "got: {colors:?}");
    for color in ["244,11,0", "11,244,0", "0,244,11", "0,11,244"] {
      assert!(colors.contains(color), "missing {color} in {colors:?}");
    }
  }

  /// A vertex color may be a list of directives (`{Specularity[…], color}`);
  /// its color is the one used.
  #[test]
  fn tube_vertex_colors_accept_directive_lists() {
    let tube = |colors: &str| {
      svg(&format!(
        "Graphics3D[{{Specularity[White, 100], Tube[{{{{0, 0, 0}}, \
         {{1, 1, 1}}}}, 0.1, VertexColors -> {colors}]}}]"
      ))
    };
    assert_eq!(
      tube("{{Specularity[White, 100], Red}, {Specularity[White, 100], Blue}}"),
      tube("{Red, Blue}")
    );
  }

  /// Options after the curve are not mistaken for the radius: a
  /// `VertexColors` tube without one keeps the default thickness.
  /// Regression: the rule was read as a radius of zero, so nothing was
  /// drawn.
  #[test]
  fn tube_option_is_not_read_as_the_radius() {
    let r = 0.0035 * 3f64.sqrt() * 4.0;
    let range = "PlotRange -> {{-2, 2}, {-2, 2}, {-2, 2}}";
    let tube = |radius: &str| {
      svg(&format!(
        "Graphics3D[Tube[{{{{0, 0, 0}}, {{1, 0, 0}}}}{radius}, \
         VertexColors -> {{Red, Blue}}], {range}]"
      ))
    };
    assert!(polygon_count(&tube("")) > 100);
    assert_eq!(tube(""), tube(&format!(", {r}")));
  }

  /// A `VertexColors` list that does not match the points is ignored, as
  /// for a line: the tube keeps its plain color.
  #[test]
  fn tube_vertex_colors_of_the_wrong_length_are_ignored() {
    let tube = |opts: &str| {
      svg(&format!(
        "Graphics3D[{{Orange, Tube[{{{{0, 0, 0}}, {{1, 1, 1}}}}, 0.1{opts}]}}]"
      ))
    };
    assert_eq!(tube(", VertexColors -> {Red, Green, Blue}"), tube(""));
  }

  /// Every edge of a wireframe drawn as separate radius-less tubes (here
  /// through a `GraphicsComplex`) is equally thick, however long it is.
  /// Regression: each two-point tube took a hundredth of its own length,
  /// so the edges of a tube icosahedron came out as hairlines.
  #[test]
  fn tube_default_radius_is_shared_by_tubes_of_any_length() {
    let r = 0.0035 * 3f64.sqrt() * 4.0;
    let range = "PlotRange -> {{-2, 2}, {-2, 2}, {-2, 2}}";
    let pts = "{{0, 0, 0}, {1.5, 0, 0}, {0, 0.2, 0}}";
    assert_eq!(
      svg(&format!(
        "Graphics3D[GraphicsComplex[{pts}, Tube[{{{{1, 2}}, {{1, 3}}}}]], \
         {range}]"
      )),
      svg(&format!(
        "Graphics3D[GraphicsComplex[{pts}, \
         {{Tube[{{1, 2}}, {r}], Tube[{{1, 3}}, {r}]}}], {range}]"
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
         EdgeForm[], Cone[{{{{{b}, 0, 0}}, {{1, 0, 0}}}}, {r}]}}, {range}]",
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

  /// Regression: a `Row` label holding a fraction (a quotient, or a string
  /// whose inline box notation stacks one) was broken over several lines of
  /// text — `x = a` above `b`, or the fraction flattened to `a/b` with the
  /// row pushed a line up onto its neighbor. It is laid out as boxes, the
  /// fraction stacked in line with the row's other parts, as in `Graphics`.
  #[test]
  fn row_text_label_stacks_its_fraction_inline() {
    for label in [
      "Row[{\"k = \", u/v}]",
      "Style[Row[{\"k = \", \"\\!\\(\\*FractionBox[\\(u\\), \\(v\\)]\\)\"}], 20, Bold]",
    ] {
      let s = svg(&format!("Graphics3D[{{Text[{label}, {{0, 0, 0}}]}}]"));
      // One nested picture: the lead-in, then `u` over a bar over `v`.
      let pic =
        &s[s.find("<svg x=").unwrap_or_else(|| panic!("{label}: {s}"))..];
      let k = pic
        .find(">k = </text>")
        .unwrap_or_else(|| panic!("{label}: {s}"));
      let u = pic
        .find(">u</text>")
        .unwrap_or_else(|| panic!("{label}: {s}"));
      let bar = pic.find("<line ").unwrap_or_else(|| panic!("{label}: {s}"));
      let v = pic
        .find(">v</text>")
        .unwrap_or_else(|| panic!("{label}: {s}"));
      assert!(k < u && u < bar && bar < v, "{label}: {s}");
      assert!(!s.contains("<tspan x="), "{label}: {s}");
    }
    let bold =
      svg("Graphics3D[{Text[Style[Row[{\"k = \", u/v}], Bold], {0, 0, 0}]}]");
    assert!(bold.contains("font-weight=\"bold\""), "{bold}");
  }

  /// An accent over a label's base (`\!\(\*OverscriptBox[\(T\), \(^\)]\)`)
  /// is drawn as its combining mark on the base (`T̂`, `x̃`), not as a
  /// superscript beside it; a non-accent overscript is stacked above it.
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
    assert!(s.contains("data-stack=\"over\""), "got: {s}");
    assert!(!s.contains("baseline-shift=\"super\""), "got: {s}");
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
    // `Scaled[0.1]` is a tenth of the *padded* width: 20 / 0.8 = 25
    // across (wolframscript's `AbsoluteOptions` of a unit `Cuboid[]` under
    // `PlotRangePadding -> Scaled[0.1]` reports a padding of 0.125).
    assert!(
      close(span("Scaled[0.1]"), 7.0 / 12.5),
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

  /// A label is cut off by the box the scene displays, which is the plot
  /// range widened by its padding: `Show` inheriting the padded automatic
  /// range of a `ParametricPlot3D` keeps a label set just off the surface,
  /// and drops one further out. With no padding the range itself cuts.
  #[test]
  fn text_is_cut_off_by_the_padded_box() {
    let labels = |options: &str| {
      let s = svg(&format!(
        "Show[{{ParametricPlot3D[{{(3 + Cos[v]) Cos[u], (3 + Cos[v]) Sin[u], \
         Sin[v]}}, {{u, 0, 2 Pi}}, {{v, 0, 2 Pi}}{options}], \
         Graphics3D[{{Text[\"Qa\", {{4.2, 0, 0}}], Text[\"Qb\", {{0, 0, 1.08}}], \
         Text[\"Qc\", {{0, 0, 1.3}}], Text[\"Qd\", {{4.5, 0, 0}}]}}]}}]"
      ));
      ["Qa", "Qb", "Qc", "Qd"].map(|l| s.contains(&format!(">{l}<")))
    };
    assert_eq!(labels(""), [true, true, false, false]);
    assert_eq!(
      labels(", PlotRangePadding -> None"),
      [false, false, false, false]
    );
  }

  /// An axis given an interval of no extent (`{0, 0}`) is framed by the
  /// contents, while the other axes keep their explicit intervals. The
  /// whole `PlotRange` used to be dropped, so every scene filled its
  /// frame regardless of the fixed ranges.
  #[test]
  fn an_axis_interval_without_extent_is_automatic() {
    let scene = |range: &str| {
      svg(&format!(
        "Graphics3D[Cylinder[{{{{0, 0, 0}}, {{0, 1, 0}}}}, 2], \
         PlotRange -> {range}]"
      ))
    };
    assert_eq!(
      scene("{{-5, 5}, {0, 0}, {-5, 5}}"),
      scene("{{-5, 5}, Automatic, {-5, 5}}")
    );
    assert_ne!(
      scene("{{-5, 5}, {0, 0}, {-5, 5}}"),
      scene("Automatic"),
      "the explicit intervals of the other axes must still frame the scene"
    );
    // A reversed interval is still not a frame at all.
    assert_eq!(scene("{{-5, 5}, {1, 0}, {-5, 5}}"), scene("Automatic"));
  }

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

  // A point at the corner of a solid turned away from the viewer is
  // hidden by the faces meeting there, which lie wholly in front of it
  // (wolframscript draws no dot over them). A point on the corner nearest
  // the viewer, or inside a face, is still drawn over the faces.
  #[test]
  fn a_point_on_a_far_corner_is_hidden_by_the_faces_in_front() {
    for (corner, view) in [
      ("{0, 0, 0}", ""),
      ("{1, 0, 0}", ", ViewPoint -> {-2, -1.5, 1.2}"),
    ] {
      let s = svg(&format!(
        "Graphics3D[{{Cuboid[{{0, 0, 0}}, {{1, 1, 1}}], PointSize[0.05], \
         Red, Point[{corner}], Green, Point[{{0.5, 0, 0.5}}]}}, \
         Boxed -> False{view}]"
      ));
      let lines: Vec<&str> = s.lines().collect();
      let attr = |l: &str, name: &str| -> f64 {
        let rest =
          &l[l.find(&format!(" {name}=\"")).unwrap() + name.len() + 3..];
        rest[..rest.find('"').unwrap()].parse().unwrap()
      };
      let dot = |color: &str| {
        let n = lines
          .iter()
          .position(|l| l.starts_with("<circle") && l.contains(color))
          .unwrap();
        (n, attr(lines[n], "cx"), attr(lines[n], "cy"))
      };
      // The draw positions of the faces with a corner at, or containing,
      // a point on screen.
      let faces = |x: f64, y: f64, inside: bool| -> Vec<usize> {
        lines
          .iter()
          .enumerate()
          .filter(|(_, l)| l.starts_with("<polygon"))
          .filter(|(_, l)| {
            let pts: Vec<(f64, f64)> = l
              .split("points=\"")
              .nth(1)
              .unwrap()
              .split('"')
              .next()
              .unwrap()
              .split(' ')
              .map(|p| {
                let (a, b) = p.split_once(',').unwrap();
                (a.parse().unwrap(), b.parse().unwrap())
              })
              .collect();
            if !inside {
              return pts.iter().any(|p| (p.0 - x).hypot(p.1 - y) < 1.0);
            }
            let mut odd = false;
            for i in 0..pts.len() {
              let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
              if (a.1 > y) != (b.1 > y)
                && x < a.0 + (y - a.1) * (b.0 - a.0) / (b.1 - a.1)
              {
                odd = !odd;
              }
            }
            odd
          })
          .map(|(n, _)| n)
          .collect()
      };
      let (red, rx, ry) = dot("rgb(255,0,0)");
      assert!(
        faces(rx, ry, false).iter().any(|&f| f > red),
        "far corner dot drawn over all faces at it: {s}"
      );
      let (green, gx, gy) = dot("rgb(0,255,0)");
      let under = faces(gx, gy, true);
      assert_ne!(under, Vec::<usize>::new());
      assert!(
        under.iter().all(|&f| f < green),
        "face center dot hidden: {s}"
      );
    }
    let s = svg(
      "Graphics3D[{Cuboid[{0, 0, 0}, {1, 1, 1}], PointSize[0.05], Red, \
       Point[{1, 0, 1}]}, Boxed -> False]",
    );
    assert!(s.find("<circle").unwrap() > s.rfind("<polygon").unwrap());
  }

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
      .filter(|(_, l)| l.starts_with("<polygon points=\"180.0,6.1 "))
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
        l.starts_with("<polygon points=\"134.1,170.5 ")
          && l.contains(" 245.4,196.0 247.0,32.6 ")
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

  /// Every corner of the `<polygon>` fills of an SVG.
  fn polygon_points(svg: &str) -> Vec<Vec<(f64, f64)>> {
    svg
      .lines()
      .filter(|l| l.starts_with("<polygon points=\""))
      .map(|l| {
        l["<polygon points=\"".len()..]
          .split('"')
          .next()
          .unwrap()
          .split(' ')
          .map(|p| {
            let (x, y) = p.split_once(',').unwrap();
            (x.parse().unwrap(), y.parse().unwrap())
          })
          .collect()
      })
      .collect()
  }

  // The facets of a smooth solid drawn without edges have only internal
  // cuts. Their fills met edge to edge, so at a low resolution the
  // anti-aliased cuts let what lies behind show through as a lattice of
  // seams across the surface. Each fill now reaches a little into the
  // neighbors it shares a cut with, but never past the silhouette.
  #[test]
  fn smooth_solid_facets_overlap_across_shared_cuts() {
    for solid in ["Cylinder[]", "Cone[]", "Sphere[]"] {
      let opaque = polygon_points(&svg(&format!(
        "Graphics3D[{{EdgeForm[], {solid}}}, Boxed -> False]"
      )));
      // A widened edge adds two corners. Most facets share an edge with a
      // neighbor; those on the silhouette keep that edge where it is.
      let widened = opaque.iter().filter(|p| p.len() > 3).count();
      let rim = opaque
        .iter()
        .filter(|p| p.len() == 5 || p.len() == 7)
        .count();
      assert!(
        widened * 2 > opaque.len(),
        "{solid}: {widened}/{}",
        opaque.len()
      );
      assert!(rim > 0, "{solid}");
      // An opaque sphere is tessellated about the line of sight, so only
      // the other solids outline the same facets when nearly opaque.
      if solid == "Sphere[]" {
        continue;
      }
      // Nearly opaque facets are not widened, and outline the same solid.
      let plain = polygon_points(&svg(&format!(
        "Graphics3D[{{EdgeForm[], Opacity[0.999], {solid}}}, Boxed -> False]"
      )));
      assert!(plain.iter().all(|p| p.len() == 3), "{solid}");
      let extent = |polys: &[Vec<(f64, f64)>]| {
        let pts = polys.iter().flatten();
        let xs = pts.clone().map(|p| p.0);
        let ys = pts.map(|p| p.1);
        [
          xs.clone().fold(f64::MAX, f64::min),
          xs.fold(f64::MIN, f64::max),
          ys.clone().fold(f64::MAX, f64::min),
          ys.fold(f64::MIN, f64::max),
        ]
      };
      // Each band reaches into the neighboring facet only.
      let (a, b) = (extent(&opaque), extent(&plain));
      for k in 0..4 {
        assert!((a[k] - b[k]).abs() <= 0.11, "{solid}: {a:?} vs {b:?}");
      }
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

  // The symbolic result carries the same mesh the plot draws (15 lines of
  // each parameter by default, `n` with `Mesh -> n`, plus the edges), so a
  // `Show` of the plot keeps it.
  #[test]
  fn parametric_surface_result_carries_its_mesh() {
    let surface = "ParametricPlot3D[{u, v, u v}, {u, 0, 1}, {v, 0, 2}";
    let lines = |opts: &str| {
      interpret(&format!(
        "Map[Length, Cases[{surface}{opts}], Line[l_] :> l, Infinity]]"
      ))
      .unwrap()
    };
    assert_eq!(lines(""), "{34}");
    assert_eq!(lines(", Mesh -> 3"), "{10}");
    assert_eq!(lines(", Mesh -> {2, 4}"), "{10}");
    assert_eq!(lines(", Mesh -> None"), "{}");
    assert_eq!(lines(", MeshStyle -> None"), "{}");
  }

  #[test]
  fn show_of_parametric_surface_keeps_its_mesh() {
    let surface = "ParametricPlot3D[{u, v, u v}, {u, 0, 1}, {v, 0, 2}";
    let mesh_lines = |s: &str| {
      s.split("<line ")
        .skip(1)
        .filter(|rest| {
          let tag = &rest[..rest.find('>').unwrap()];
          tag.contains("stroke=\"rgb(0,0,0)\"") && tag.contains("opacity=")
        })
        .count()
    };
    let meshed = svg(&format!("Show[{surface}], ViewPoint -> {{0, -2, 1}}]"));
    let bare = svg(&format!(
      "Show[{surface}, Mesh -> None], ViewPoint -> {{0, -2, 1}}]"
    ));
    assert!(mesh_lines(&meshed) > 200, "{}", mesh_lines(&meshed));
    assert_eq!(mesh_lines(&bare), 0);
  }

  // Like wolframscript, the result pins its range to the sampled surface's
  // extent; an explicit `PlotRange` is kept, spelled out per axis (a single
  // pair is the vertical range, recorded for every axis).
  #[test]
  fn parametric_surface_result_pins_its_plot_range() {
    assert_eq!(
      interpret(
        "PlotRange /. Options[ParametricPlot3D[{u, v, u v}, {u, 0, 1}, \
         {v, 0, 2}]]"
      )
      .unwrap(),
      "{{0., 1.}, {0., 2.}, {0., 2.}}"
    );
    assert_eq!(
      interpret(
        "PlotRange /. Options[ParametricPlot3D[{u, v, u v}, {u, 0, 1}, \
         {v, 0, 2}, PlotRange -> {0, 1}]]"
      )
      .unwrap(),
      "{{0, 1}, {0, 1}, {0, 1}}"
    );
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

  /// The arrowhead `<polygon>`s of a scene: the filled, unstroked ones
  /// (faces are stroked with a hairline), each with its point list.
  fn head_polygons(svg: &str) -> Vec<&str> {
    svg
      .lines()
      .filter(|l| l.starts_with("<polygon points=\"") && !l.contains("stroke"))
      .map(|l| l["<polygon points=\"".len()..].split('"').next().unwrap())
      .collect()
  }

  /// The area of the first filled arrowhead `<polygon>`, by the shoelace
  /// formula.
  fn head_area(svg: &str) -> f64 {
    let pts = head_polygons(svg)[0];
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

  /// The first point of every arrowhead `<polygon>` (each head's outline
  /// starts at its tip).
  fn head_tips(svg: &str) -> Vec<(f64, f64)> {
    head_polygons(svg)
      .into_iter()
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

  /// The screen midpoint of every `<line>` piece stroked in `color`, with
  /// its position among the lines of `svg`.
  /// The line pieces drawn in `color`: each `<line>`, and each segment
  /// between consecutive points of a `<polyline>` (opaque pieces drawn one
  /// after another are joined into one), with the index of the SVG line
  /// that draws it and its midpoint.
  fn line_pieces(svg: &str, color: &str) -> Vec<(usize, (f64, f64))> {
    let attr = |l: &str, name: &str| -> f64 {
      let key = format!(" {name}=\"");
      let rest = &l[l.find(&key).unwrap() + key.len()..];
      rest[..rest.find('"').unwrap()].parse().unwrap()
    };
    let stroke = format!("stroke=\"{color}\"");
    let mut pieces = Vec::new();
    for (i, l) in svg.lines().enumerate() {
      if !l.contains(&stroke) {
        continue;
      }
      if l.starts_with("<line ") {
        let x = f64::midpoint(attr(l, "x1"), attr(l, "x2"));
        let y = f64::midpoint(attr(l, "y1"), attr(l, "y2"));
        pieces.push((i, (x, y)));
      } else if l.starts_with("<polyline ") {
        let pts = l["<polyline points=\"".len()..].split('"').next().unwrap();
        let pts: Vec<(f64, f64)> = pts
          .split(' ')
          .map(|xy| {
            let mut it = xy.split(',').map(|v| v.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
          })
          .collect();
        for w in pts.windows(2) {
          pieces.push((
            i,
            (f64::midpoint(w[0].0, w[1].0), f64::midpoint(w[0].1, w[1].1)),
          ));
        }
      }
    }
    pieces
  }

  // An explicit `PlotRange` cuts a 3D arrow off at the box like a `Line`:
  // the shaft stops at the box's faces and a head whose tip falls outside
  // the box is not drawn, while a head inside it keeps its place and size.
  #[test]
  fn plot_range_clips_3d_arrow() {
    let scene = |prims: &str| {
      svg(&format!(
        "Graphics3D[{{Red, Arrowheads[{{-0.1, 0.1}}], {prims}}}, \
         PlotRange -> {{{{-1, 1}}, {{-1, 1}}, {{-1, 1}}}}]"
      ))
    };
    let extent = |s: &str| {
      let xs: Vec<f64> = line_pieces(s, "rgb(255,0,0)")
        .into_iter()
        .map(|(_, (x, _))| x)
        .collect();
      assert!(!xs.is_empty(), "{s}");
      let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
      let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
      (lo, hi)
    };

    // Both ends outside: no heads, and the shaft spans just the box.
    let through = scene("Arrow[{{-3, 0, 0}, {3, 0, 0}}]");
    assert!(head_tips(&through).is_empty(), "{through}");
    let line = scene("Line[{{-1, 0, 0}, {1, 0, 0}}]");
    let (a, b) = (extent(&through), extent(&line));
    assert!(
      (a.0 - b.0).abs() < 2.0 && (a.1 - b.1).abs() < 2.0,
      "{a:?} vs {b:?}"
    );

    // Tail outside, tip inside: only the head at the tip remains, exactly
    // where the arrow cut at the box puts it.
    let half = scene("Arrow[{{0, 0, -3}, {0, 0, 0.5}}]");
    let cut = scene("Arrow[{{0, 0, -1}, {0, 0, 0.5}}]");
    let tips = head_tips(&half);
    assert_eq!(tips.len(), 1, "{half}");
    let expected = head_tips(&cut)
      .into_iter()
      .min_by(|p, q| p.1.total_cmp(&q.1))
      .unwrap();
    assert!(
      (tips[0].0 - expected.0).abs() < 0.5
        && (tips[0].1 - expected.1).abs() < 0.5,
      "{tips:?} vs {expected:?}"
    );
    let tip_only = scene("Arrowheads[0.1], Arrow[{{0, 0, -1}, {0, 0, 0.5}}]");
    let (got, want) = (head_area(&half), head_area(&tip_only));
    assert!((got - want).abs() < 0.05 * want, "{got} vs {want}");
  }

  // An opaque solid hides the part of a 3D arrow inside or behind it: the
  // shaft is cut into depth-sorted pieces like a `Line`, so the sphere's
  // front faces are painted over the pieces they cover.
  #[test]
  fn opaque_sphere_hides_the_arrow_shaft_through_it() {
    let s = svg(
      "Graphics3D[{Red, Thickness[0.01], Arrow[{{-3, 0, 0}, {3, 0, 0}}], \
       White, Sphere[{0, 0, 0}, 1]}, Boxed -> False, ViewPoint -> {2, 2, 2}]",
    );
    let lines: Vec<&str> = s.lines().collect();
    let last_face = lines
      .iter()
      .rposition(|l| l.starts_with("<polygon") && l.contains("stroke"))
      .expect("sphere faces");
    // The sphere's outline on screen, from the extent of its faces.
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for l in lines
      .iter()
      .filter(|l| l.contains("stroke") && l.starts_with("<polygon"))
    {
      let pts = l["<polygon points=\"".len()..].split('"').next().unwrap();
      for xy in pts.split(' ') {
        let mut it = xy.split(',').map(|v| v.parse::<f64>().unwrap());
        let (x, y) = (it.next().unwrap(), it.next().unwrap());
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
      }
    }
    let center = (f64::midpoint(lo.0, hi.0), f64::midpoint(lo.1, hi.1));
    let radius = (hi.0 - lo.0) / 2.0;
    let pieces = line_pieces(&s, "rgb(255,0,0)");
    // The shaft is drawn on both sides of the sphere.
    assert!(
      pieces.iter().any(|&(_, (x, _))| x < center.0 - radius),
      "{s}"
    );
    assert!(
      pieces.iter().any(|&(_, (x, _))| x > center.0 + radius),
      "{s}"
    );
    // No piece over the middle of the sphere is painted on top of it.
    for &(i, (x, y)) in &pieces {
      let d = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
      assert!(i < last_face || d > 0.8 * radius, "piece at {x},{y}: {s}");
    }
  }

  // A 3D arrowhead inside an opaque solid is hidden by it as well.
  #[test]
  fn opaque_sphere_hides_an_arrowhead_inside_it() {
    let s = svg(
      "Graphics3D[{Red, Arrowheads[0.05], Arrow[{{-3, 0, 0}, {0, 0, 0}}], \
       White, Sphere[{0, 0, 0}, 1]}, Boxed -> False]",
    );
    let lines: Vec<&str> = s.lines().collect();
    let head = lines
      .iter()
      .position(|l| {
        l.starts_with("<polygon") && l.contains("fill=\"rgb(255,0,0)\"")
      })
      .expect("arrowhead");
    let last_face = lines
      .iter()
      .rposition(|l| l.starts_with("<polygon") && l.contains("stroke"))
      .expect("sphere faces");
    assert!(head < last_face, "{s}");
  }

  // A long straight segment is cut into pieces a few pixels long, so where
  // it passes into a surface it disappears close to the crossing rather
  // than up to an eighth of its length away.
  #[test]
  fn long_3d_line_is_cut_finely_for_depth_sorting() {
    let s =
      svg("Graphics3D[{Red, Line[{{-3, 0, 0}, {3, 0, 0}}]}, Boxed -> False]");
    assert!(line_pieces(&s, "rgb(255,0,0)").len() > 32, "{s}");
    // A short one is not cut at all.
    let s = svg(
      "Graphics3D[{Red, Line[{{0, 0, 0}, {0.01, 0, 0}}], \
       Point[{{-3, -3, -3}, {3, 3, 3}}]}, Boxed -> False]",
    );
    assert_eq!(line_pieces(&s, "rgb(255,0,0)").len(), 1, "{s}");
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

  // An opaque face lying exactly under a later opaque one is never seen, so
  // it is not drawn at all: left in, its color bleeds through the
  // anti-aliased seams between the triangles of the face on top. The copy
  // may list the same corners from another start or the other way round,
  // and may be moved onto the face by a `Translate`.
  #[test]
  fn opaque_face_under_an_identical_face_is_not_drawn() {
    let alone = svg(
      "Graphics3D[{Yellow, Polygon[{{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, \
       {1, 2, 0}}]}, Boxed -> False]",
    );
    for under in [
      "Polygon[{{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {1, 2, 0}}]",
      "Polygon[{{3, 2, 0}, {1, 2, 0}, {0, 0, 0}, {2, 0, 0}}]",
      "Polygon[{{1, 2, 0}, {3, 2, 0}, {2, 0, 0}, {0, 0, 0}}]",
      "Translate[Polygon[{{0, 0, 5}, {2, 0, 5}, {3, 2, 5}, {1, 2, 5}}], \
       {0, 0, -5}]",
      "Translate[Polygon[{{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {1, 2, 0}}], \
       0*{1, 2, 3}]",
    ] {
      let stacked = svg(&format!(
        "Graphics3D[{{Cyan, {under}, Orange, {under}, Yellow, \
         Polygon[{{{{0, 0, 0}}, {{2, 0, 0}}, {{3, 2, 0}}, {{1, 2, 0}}}}]}}, \
         Boxed -> False]"
      ));
      assert_eq!(stacked, alone, "{under}");
    }
  }

  // Faces that only overlap, a translucent face on top, and a copy with an
  // outline the face on top does not repeat are all still drawn.
  #[test]
  fn faces_that_are_not_wholly_covered_are_kept() {
    let quad = "Polygon[{{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {1, 2, 0}}]";
    let one = polygon_count(&svg(&format!(
      "Graphics3D[{{Yellow, {quad}}}, Boxed -> False]"
    )));
    for scene in [
      // Shifted within the plane.
      format!("{{Cyan, Translate[{quad}, {{0.5, 0, 0}}], Yellow, {quad}}}"),
      // A different polygon on the same corners but one.
      format!(
        "{{Cyan, Polygon[{{{{0, 0, 0}}, {{2, 0, 0}}, {{3, 2, 0}}, \
         {{1, 3, 0}}}}], Yellow, {quad}}}"
      ),
      // The face on top lets the lower one show through.
      format!("{{Cyan, {quad}, Opacity[0.5], Yellow, {quad}}}"),
    ] {
      let n =
        polygon_count(&svg(&format!("Graphics3D[{scene}, Boxed -> False]")));
      assert!(n > one, "{scene}: {n} vs {one}");
    }
    let outlined = svg(&format!(
      "Graphics3D[{{EdgeForm[{{Thick, Red}}], Cyan, {quad}, EdgeForm[], \
       Yellow, {quad}}}, Boxed -> False]"
    ));
    assert!(outlined.contains("stroke=\"rgb(255,0,0)\""), "{outlined}");
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
  // others keep the extent of the contents, padded like any automatic
  // range (wolframscript's `AbsoluteOptions` reports a padding of
  // `{{0, 0}, {1/24, 1/24}, {1/16, 1/16}}`).
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
           PlotRange -> {{0, 4}, {0, 2}, {0, 3}}, \
           PlotRangePadding -> {0, Scaled[0.02], Scaled[0.02]}]"
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

  // A font size in `TicksStyle` sizes the tick labels of the axes it
  // styles, overriding a `LabelStyle` size.
  #[test]
  fn ticks_style_font_size_sizes_tick_labels() {
    let s = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> True, \
       TicksStyle -> Directive[17, Gray]]",
    );
    let gray17 = "font-size=\"17\" fill=\"rgb(128,128,128)\"";
    assert!(s.matches(gray17).count() >= 6, "{s}");
    assert!(!s.contains("font-size=\"10\""), "{s}");

    let s = svg(
      "Graphics3D[Line[{{0, 0, 0}, {4, 1, 1}}], Axes -> True, \
       LabelStyle -> 12, TicksStyle -> {Red, 19, Blue}]",
    );
    assert!(s.contains("font-size=\"12\" fill=\"rgb(255,0,0)\""), "{s}");
    assert!(s.contains("font-size=\"19\""), "{s}");
    assert!(s.contains("font-size=\"12\" fill=\"rgb(0,0,255)\""), "{s}");
  }

  // `Plot3D` styles its tick labels with `TicksStyle` too.
  #[test]
  fn plot3d_ticks_style_colors_and_sizes_tick_labels() {
    let s = svg(
      "Plot3D[x y, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 5, Mesh -> None, \
       TicksStyle -> Directive[15, Red]]",
    );
    let red15 = "font-size=\"15\" fill=\"rgb(255,0,0)\"";
    assert!(s.matches(red15).count() >= 6, "{s}");
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

  /// Overlapping translucent halves would darken the diagonal, so they
  /// do not widen.
  #[test]
  fn translucent_faces_keep_three_corners() {
    let s = svg(&format!(
      "Graphics3D[{{Opacity[0.5], {SQUARE}}}, Boxed -> False]"
    ));
    // A translucent face's pieces are filled as one `<path>` instead.
    let counts = polygon_point_counts(&s);
    assert!(counts.iter().all(|&n| n == 3), "{s}");
  }

  /// Without a drawn outline nothing would cover a band leaving a flat
  /// face, so its halves keep their exact corners.
  #[test]
  fn edgeless_flat_faces_keep_three_corners() {
    let s = svg(&format!(
      "Graphics3D[{{EdgeForm[], {SQUARE}}}, Boxed -> False]"
    ));
    assert_eq!(polygon_point_counts(&s), vec![3, 3], "{s}");
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

mod coincident_lines {
  use super::*;

  /// For every on-screen line position drawn in both colors, the color
  /// of the stroke that ends up on top. A position is a segment of a
  /// `<line>` or between consecutive points of a `<polyline>`.
  fn top_colors(svg: &str) -> Vec<String> {
    let mut last: std::collections::HashMap<String, (String, bool)> =
      std::collections::HashMap::new();
    let attr = |el: &str, name: &str| -> Option<String> {
      el.split(&format!(" {name}=\""))
        .nth(1)
        .and_then(|s| s.split('"').next())
        .map(str::to_string)
    };
    for el in svg.lines() {
      let Some(color) = attr(el, "stroke") else {
        continue;
      };
      if color != "rgb(0,0,255)" && color != "rgb(255,0,0)" {
        continue;
      }
      let segments: Vec<String> = if el.starts_with("<line ") {
        let c: Vec<String> = ["x1", "y1", "x2", "y2"]
          .iter()
          .map(|a| attr(el, a).unwrap_or_default())
          .collect();
        vec![format!("{},{} {},{}", c[0], c[1], c[2], c[3])]
      } else if el.starts_with("<polyline ") {
        let pts: Vec<&str> = el["<polyline points=\"".len()..]
          .split('"')
          .next()
          .unwrap()
          .split(' ')
          .collect();
        pts
          .windows(2)
          .map(|w| format!("{} {}", w[0], w[1]))
          .collect()
      } else {
        continue;
      };
      for key in segments {
        let entry = last.entry(key).or_insert((color.clone(), false));
        if entry.0 != color {
          entry.1 = true;
        }
        entry.0 = color.clone();
      }
    }
    last
      .into_values()
      .filter(|(_, both)| *both)
      .map(|(c, _)| c)
      .collect()
  }

  #[test]
  fn later_line_covers_an_earlier_one_at_the_same_depth() {
    // The second curve wobbles around the first by far less than a
    // pixel, so a strict depth sort would interleave the two; as with a
    // depth buffer, the later primitive must stay on top throughout.
    let out = svg(
      "Graphics3D[{Blue, Line[Table[{Cos[t], Sin[t], t/5}, {t, 0, 12, 0.1}]], \
       Red, Line[Table[{Cos[t], Sin[t], t/5} + (-1)^Round[10 t] 10^-9, \
       {t, 0, 12, 0.1}]]}, Boxed -> False]",
    );
    let tops = top_colors(&out);
    assert!(tops.len() > 50, "too few shared positions: {}", tops.len());
    assert!(
      tops.iter().all(|c| c == "rgb(255,0,0)"),
      "blue drawn over red at {} of {} positions",
      tops.iter().filter(|c| *c == "rgb(0,0,255)").count(),
      tops.len()
    );
  }
}

mod plot_label {
  use super::*;

  /// The `font-size` of the first `<text>` element of an SVG.
  fn first_text_font_size(svg: &str) -> f64 {
    let text = &svg[svg.find("<text").expect("no <text>")..];
    let start = text.find("font-size=\"").unwrap() + "font-size=\"".len();
    let end = start + text[start..].find('"').unwrap();
    text[start..end].parse().unwrap()
  }

  #[test]
  fn short_title_keeps_the_default_size() {
    let s = svg("Graphics3D[Cuboid[], PlotLabel -> \"title\"]");
    assert_eq!(first_text_font_size(&s), 16.0, "{s}");
  }

  #[test]
  fn wide_title_shrinks_to_fit_the_picture() {
    // Regression: a title wider than the picture (a long readout `Row`)
    // was set at 16pt and clipped at both edges.
    let s = svg(
      "Graphics3D[Cuboid[], ImageSize -> 300, PlotLabel -> Row[{\
       \"energy = -2.0000, kinetic = 0.0000, potential = -2.0000 units\"}]]",
    );
    let size = first_text_font_size(&s);
    assert!((9.0..16.0).contains(&size), "font-size {size}: {s}");
  }

  #[test]
  fn explicit_font_size_is_kept() {
    let long = "a long title that is far wider than the picture it labels";
    let s = svg(&format!(
      "Graphics3D[Cuboid[], ImageSize -> 200, \
       PlotLabel -> Style[\"{long}\", 20]]"
    ));
    assert_eq!(first_text_font_size(&s), 20.0, "{s}");
  }
}

mod face_form_scope {
  use super::*;

  // `FaceForm` styles only the faces that follow it: a translucent plane
  // leaves the lines, arrows and points after it opaque.
  #[test]
  fn face_form_opacity_leaves_lines_and_points_opaque() {
    let out = svg(
      "Graphics3D[{FaceForm[{Opacity[0.6]}], \
       Polygon[{{-1, 2, 0}, {1, 2, 0}, {1, -2, 0}, {-1, -2, 0}}], \
       Line[{{2, 0, -1}, {2, 0, 1}}], Arrow[{{0, 0, -2}, {0, 0, 2}}], \
       PointSize[Large], Point[{0, 0, 0}]}, Boxed -> False]",
    );
    // Only the plane is drawn translucent.
    assert_eq!(out.matches("opacity=\"0.6\"").count(), 1, "{out}");
    assert!(out.contains("<path") && out.contains("opacity=\"0.6\""));
    for tag in ["<polyline", "<circle"] {
      let el = out.split(tag).nth(1).unwrap().split('>').next().unwrap();
      assert!(!el.contains("opacity"), "{tag}: {el}");
    }
  }

  // A `FaceForm` color does not recolor the lines either, while a plain
  // color after it recolors the lines but not the faces.
  #[test]
  fn face_form_color_applies_to_faces_only() {
    let out = svg(
      "Graphics3D[{FaceForm[Red], Line[{{0, 0, 0}, {1, 1, 1}}], Blue, \
       Line[{{1, 0, 0}, {0, 1, 1}}], Cuboid[{0, 0, 0}, {1, 1, 1}]}, \
       Lighting -> {{\"Ambient\", White}}, Boxed -> False]",
    );
    // The first line keeps the default black, the second is blue.
    assert!(out.contains("stroke=\"black\""), "{out}");
    assert!(out.contains("stroke=\"rgb(0,0,255)\""), "{out}");
    // The cuboid stays red under the later plain `Blue`.
    assert!(fills(&out).contains("255,0,0"), "{:?}", fills(&out));
    assert!(!fills(&out).contains("0,0,255"), "{:?}", fills(&out));
  }

  // A `FaceForm` that names only an opacity keeps the plain color for the
  // faces.
  #[test]
  fn face_form_opacity_keeps_plain_face_color() {
    let out = svg(
      "Graphics3D[{Red, FaceForm[Opacity[0.5]], Cuboid[]}, \
       Lighting -> {{\"Ambient\", White}}, Boxed -> False]",
    );
    assert!(fills(&out).contains("255,0,0"), "{:?}", fills(&out));
    assert!(out.contains("opacity=\"0.5\""), "{out}");
  }
}

mod arrow_setback_3d {
  use super::*;

  /// The `(x1, y1, x2, y2)` of every arrow shaft or line `<polyline>`
  /// (its first and last points).
  fn shafts(svg: &str) -> Vec<(f64, f64, f64, f64)> {
    svg
      .split("<polyline points=\"")
      .skip(1)
      .map(|el| {
        let pts: Vec<(f64, f64)> = el
          .split('"')
          .next()
          .unwrap()
          .split(' ')
          .map(|p| {
            let mut it = p.split(',').map(|v| v.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
          })
          .collect();
        let (a, b) = (pts[0], pts[pts.len() - 1]);
        (a.0, a.1, b.0, b.1)
      })
      .collect()
  }

  fn length(s: (f64, f64, f64, f64)) -> f64 {
    (s.2 - s.0).hypot(s.3 - s.1)
  }

  const FRAME: &str = "Point[{{-3, -3, -3}, {3, 3, 3}}]";

  // `Arrow[pts, s]` sets both ends back by `s` along the arrow; `{s1, s2}`
  // sets the tail and the tip back separately.
  #[test]
  fn positive_setback_shortens_the_arrow() {
    let arrow = |spec: &str| {
      svg(&format!(
        "Graphics3D[{{Arrow[{{{{0, 0, -2}}, {{0, 0, 2}}}}{spec}], {FRAME}}}, \
         Boxed -> False]"
      ))
    };
    let full = length(shafts(&arrow(""))[0]);
    let half = length(shafts(&arrow(", 1"))[0]);
    let tip_only = length(shafts(&arrow(", {0, 2}"))[0]);
    assert!((half / full - 0.5).abs() < 0.05, "{half} vs {full}");
    assert!((tip_only / full - 0.5).abs() < 0.05, "{tip_only} vs {full}");
    // A setback longer than the arrow leaves nothing to draw.
    assert_eq!(shafts(&arrow(", 3")), Vec::new());
  }

  // A negative setback lengthens the arrow past both of its end points.
  #[test]
  fn negative_setback_lengthens_the_arrow() {
    let arrow = |spec: &str| {
      svg(&format!(
        "Graphics3D[{{Arrow[{{{{0, 0, -1}}, {{0, 0, 1}}}}{spec}], {FRAME}}}, \
         Boxed -> False]"
      ))
    };
    let plain = arrow("");
    let long = arrow(", -1");
    let ratio = length(shafts(&long)[0]) / length(shafts(&plain)[0]);
    assert!((ratio - 2.0).abs() < 0.15, "{ratio}");
    // It keeps its head.
    assert_eq!(polygon_count(&long), polygon_count(&plain));
  }

  /// The lowest and highest screen y any black stroke reaches.
  fn black_y_extent(svg: &str) -> (f64, f64) {
    let mut ys = Vec::new();
    for el in svg.split('<').filter(|el| el.contains("stroke=\"black\"")) {
      let attr = |name: &str| {
        el.split(&format!(" {name}=\""))
          .nth(1)
          .and_then(|v| v.split('"').next())
          .map(str::to_string)
      };
      if let Some(pts) = attr("points") {
        for p in pts.split(' ') {
          ys.push(p.split(',').nth(1).unwrap().parse::<f64>().unwrap());
        }
      }
      for name in ["y1", "y2"] {
        if let Some(v) = attr(name) {
          ys.push(v.parse::<f64>().unwrap());
        }
      }
    }
    ys.iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &y| {
        (lo.min(y), hi.max(y))
      })
  }

  // The lengthened arrow does not widen the scene's range: what reaches
  // past the box its end points fix is cut off, head and all.
  #[test]
  fn lengthened_arrow_is_cut_at_the_box() {
    let scene = |spec: &str| {
      svg(&format!(
        "Graphics3D[{{FaceForm[Opacity[0.6]], Polygon[{{{{-1, 2, 0}}, \
         {{1, 2, 0}}, {{1, -2, 0}}, {{-1, -2, 0}}}}], \
         Arrow[{{{{0, 0, -2}}, {{0, 0, 2}}}}{spec}]}}, \
         Boxed -> False, ImageSize -> {{400, 400}}]"
      ))
    };
    let plain = scene("");
    let long = scene(", -1");
    // The plane is framed the same.
    let plane = |s: &str| {
      s.split("<path d=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
    };
    assert_eq!(plane(&plain), plane(&long));
    // The shaft runs to the edges of the box, but no head is drawn.
    let (lo, hi) = black_y_extent(&long);
    let (plain_lo, plain_hi) = black_y_extent(&plain);
    assert!((lo - plain_lo).abs() < 2.0 && (hi - plain_hi).abs() < 2.0);
    assert!(plain.contains("<polygon"), "{plain}");
    assert!(!long.contains("<polygon"), "{long}");
  }

  // The setback follows the arrow through a rotation.
  #[test]
  fn setback_survives_rotate() {
    let out = svg(&format!(
      "Graphics3D[{{Rotate[Arrow[{{{{0, 0, -1}}, {{0, 0, 1}}}}, -1], Pi/2, \
       {{1, 0, 0}}], {FRAME}}}, Boxed -> False]"
    ));
    let plain = svg(&format!(
      "Graphics3D[{{Rotate[Arrow[{{{{0, 0, -2}}, {{0, 0, 2}}}}], Pi/2, \
       {{1, 0, 0}}], {FRAME}}}, Boxed -> False]"
    ));
    let (a, b) = (shafts(&out)[0], shafts(&plain)[0]);
    assert!((length(a) - length(b)).abs() < 1.0, "{a:?} vs {b:?}");
  }
}

mod tube_spline_radius {
  use super::*;

  /// Regression: a single radius written as the spline's second argument
  /// (`Tube[BSplineCurve[pts, r]]`) is the tube's radius, as a radius list
  /// there already was; Woxi drew such a tube at the thin default radius.
  #[test]
  fn radius_in_the_curve_sets_the_tube_radius() {
    let scene = |tube: &str| {
      svg(&format!(
        "Graphics3D[{{{tube}}}, PlotRange -> {{{{-3, 3}}, {{-3, 3}},          {{-3, 3}}}}]"
      ))
    };
    let pts = "{{-2, 0, 0}, {-1, 1, 0}, {1, -1, 1}, {2, 0, 0}}";
    for radius in ["0.5", "1/4", "0.1"] {
      let inside = scene(&format!("Tube[BSplineCurve[{pts}, {radius}]]"));
      let outside = scene(&format!("Tube[BSplineCurve[{pts}], {radius}]"));
      assert_eq!(inside, outside, "{radius}");
    }
    assert_ne!(
      scene(&format!("Tube[BSplineCurve[{pts}, 0.5]]")),
      scene(&format!("Tube[BSplineCurve[{pts}]]"))
    );
    // The tube's own radius wins over one in the curve.
    assert_eq!(
      scene(&format!("Tube[BSplineCurve[{pts}, 0.5], 0.2]")),
      scene(&format!("Tube[BSplineCurve[{pts}], 0.2]"))
    );
    // An option after the points is not a radius.
    assert_ne!(
      scene(&format!("Tube[BSplineCurve[{pts}, SplineDegree -> 2]]")),
      scene(&format!(
        "Tube[BSplineCurve[{pts}, SplineDegree -> 2], 0.5]"
      ))
    );
  }
}

mod surface_outlines_and_meshes {
  use super::*;

  /// The end points of every `<line>` stroked in `stroke`, as written.
  fn line_ends(svg: &str, stroke: &str) -> Vec<[(String, String); 2]> {
    svg
      .split("<line ")
      .skip(1)
      .filter_map(|l| l.split("/>").next())
      .filter(|l| l.contains(&format!("stroke=\"{stroke}\"")))
      .filter_map(|l| {
        let attr = |k: &str| -> Option<String> {
          Some(
            l.split(&format!("{k}=\""))
              .nth(1)?
              .split('"')
              .next()?
              .into(),
          )
        };
        Some([(attr("x1")?, attr("y1")?), (attr("x2")?, attr("y2")?)])
      })
      .collect()
  }

  /// The corners of every `<polygon>`.
  fn polygon_corners(svg: &str) -> Vec<Vec<(f64, f64)>> {
    svg
      .split("<polygon points=\"")
      .skip(1)
      .filter_map(|s| s.split('"').next())
      .map(|pts| {
        pts
          .split(' ')
          .filter_map(|p| {
            let (x, y) = p.split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
          })
          .collect()
      })
      .collect()
  }

  // Wolfram outlines a `BSplineSurface` along its open rim in its
  // `EdgeForm`, and `EdgeForm[None]` leaves it bare. Woxi drew no rim at
  // all, so the curved edges of the surface faded into the background.
  #[test]
  fn bspline_surface_rim_follows_edge_form() {
    let scene = |edge: &str| {
      svg(&format!(
        "Graphics3D[{{{edge}BSplineSurface[Table[{{i, j, Sin[i] Cos[j]}}, \
         {{i, 0, 3}}, {{j, 0, 3}}]]}}, Boxed -> False]"
      ))
    };
    let red = scene("EdgeForm[Red], ");
    assert!(red.contains("rgb(255,0,0)"), "a red rim: {red}");
    let bare = scene("EdgeForm[None], ");
    assert!(!bare.contains("rgb(255,0,0)"));
    assert_ne!(scene(""), bare, "the default EdgeForm draws a rim");
  }

  // A torus is closed: it has no rim to outline.
  #[test]
  fn closed_surfaces_have_no_rim() {
    let red = svg("Graphics3D[{EdgeForm[Red], Torus[]}, Boxed -> False]");
    assert!(!red.contains("rgb(255,0,0)"), "{red}");
  }

  // A contour level passing exactly through a sample point collapsed the
  // marching-tetrahedra crossings onto that point. The zero-area triangles
  // left behind were drawn as stray strokes across the surface.
  #[test]
  fn contour_surfaces_have_no_degenerate_triangles() {
    let s = svg(
      "ListContourPlot3D[Table[x^2 + y^2 + z^2, {x, -1, 1, 0.25}, \
       {y, -1, 1, 0.25}, {z, -1, 1, 0.25}], Contours -> {0.5}, Mesh -> None]",
    );
    let corners = polygon_corners(&s);
    assert!(corners.len() > 100, "{s}");
    let degenerate = corners
      .iter()
      .filter(|p| {
        p.len() == 3 && (p[0] == p[1] || p[1] == p[2] || p[0] == p[2])
      })
      .count();
    assert_eq!(degenerate, 0, "{s}");
  }

  // With a `RegionFunction`, the default mesh lines of Plot3D run on to the
  // rim where the region cuts the surface, as in Wolfram, instead of
  // stopping at the last sample inside the region.
  #[test]
  fn plot3d_mesh_lines_run_to_the_region_cut() {
    let s = svg(
      "Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}, \
       RegionFunction -> (#1^2 + #2^2 <= 4 &), PlotPoints -> 20, \
       MaxRecursion -> 1]",
    );
    let rim: std::collections::HashSet<(String, String)> =
      line_ends(&s, "rgb(0,0,0)").into_iter().flatten().collect();
    let meeting = line_ends(&s, "#000000a0")
      .into_iter()
      .flatten()
      .filter(|p| rim.contains(p))
      .count();
    // About 60 mesh lines end on the rim; only the few whose last sample
    // happened to lie on the circle met it before.
    assert!(meeting >= 40, "{meeting} mesh line ends on the rim: {s}");
  }

  // Wolfram's ListPlot3D cuts each grid cell along its diagonal from
  // `(i, j)` to `(i + 1, j + 1)`. From the default `ViewPoint ->
  // {1.3, -2.4, 2}` that diagonal joins the leftmost and the rightmost
  // corner of a cell on screen; Woxi cut along the other one, from the
  // topmost to the bottommost corner.
  #[test]
  fn list_plot3d_cuts_cells_along_the_rising_diagonal() {
    let s = svg("ListPlot3D[{{0, 0}, {0, 1}}, Mesh -> All, MeshStyle -> Blue]");
    let num = |v: &String| v.parse::<f64>().unwrap();
    let lines: Vec<[(f64, f64); 2]> = line_ends(&s, "rgb(0,0,255)")
      .iter()
      .map(|[a, b]| [(num(&a.0), num(&a.1)), (num(&b.0), num(&b.1))])
      .collect();
    assert_eq!(lines.len(), 5, "four sides and a diagonal: {s}");
    let xs: Vec<f64> = lines.iter().flatten().map(|p| p.0).collect();
    let left = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let right = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
      lines.iter().any(|[a, b]| {
        let mut ends = [a.0, b.0];
        ends.sort_by(f64::total_cmp);
        ends == [left, right]
      }),
      "{s}"
    );
  }

  // The default mesh lines of ListPlot3D lie on the triangles the surface
  // is drawn as: a line crossing a cell bends where it crosses the cell's
  // diagonal. Drawn straight across the cell, a line dipped under one of
  // the two triangles, which hid stretches of it.
  #[test]
  fn list_plot3d_mesh_lines_bend_at_the_cell_diagonal() {
    let s = svg("ListPlot3D[{{0, 0}, {0, 1}}]");
    // 17 lines each way, the 15 inner ones in two pieces each.
    assert_eq!(s.matches("stroke=\"#000000a0\"").count(), 64, "{s}");
  }

  // ParametricPlot3D normalizes its box to ±BoxRatios and looks at it
  // from |ViewPoint| times that box, so scaling the surface changes
  // nothing on screen — its shading included. The lights used to judge
  // which side of a facet faces the eye from the default camera's eye,
  // which stood inside the box of a torus of radius 3 and lit much of it
  // from behind (dark, toothed patches).
  #[test]
  fn parametric_plot3d_shading_does_not_depend_on_scale() {
    let stops = |k: u32| -> Vec<String> {
      let s = svg(&format!(
        "ParametricPlot3D[{k} {{(2 + Cos[v]) Cos[u], (2 + Cos[v]) Sin[u], \
         Sin[v]}}, {{u, 0, 2 Pi}}, {{v, 0, 2 Pi}}, PlotPoints -> 10, \
         MaxRecursion -> 0, Mesh -> None, Axes -> False]"
      ));
      s.split("stop-color=\"")
        .skip(1)
        .filter_map(|r| r.split('"').next().map(str::to_string))
        .collect()
    };
    let (small, large) = (stops(1), stops(10));
    assert_ne!(small, Vec::<String>::new());
    assert_eq!(small, large);
  }

  // ParametricPlot3D draws each grid cell as two triangles cut along the
  // diagonal from (i + 1, j) to (i, j + 1); a mesh line between two grid
  // rows bends where it crosses that diagonal so it stays on the
  // triangles. Straight across a twisted cell (a Möbius strip) it dipped
  // under one of them, which hid stretches of it.
  #[test]
  fn parametric_plot3d_mesh_lines_bend_at_the_cell_diagonal() {
    let s = svg(
      "ParametricPlot3D[{u, v, u v}, {u, 0, 1}, {v, 0, 1}, PlotPoints -> 2, \
       MaxRecursion -> 0]",
    );
    assert_eq!(polygon_count(&s), 2, "{s}");
    // 17 lines each way, the 15 inner ones in two pieces each.
    assert_eq!(s.matches("stroke=\"#000000a0\"").count(), 64, "{s}");
  }

  // `PlotPoints -> n` with `MaxRecursion -> r` samples ParametricPlot3D
  // more finely than `PlotPoints` alone, up to `(n - 1) 2^r` cells per
  // direction, as Wolfram's adaptive sampler may. `MaxRecursion` was
  // ignored whenever `PlotPoints` was given.
  #[test]
  fn parametric_plot3d_max_recursion_refines_the_grid() {
    let count = |r: u32| {
      polygon_count(&svg(&format!(
        "ParametricPlot3D[{{Cos[u], Sin[u], v}}, {{u, 0, 2}}, {{v, 0, 1}}, \
         PlotPoints -> 10, MaxRecursion -> {r}, Mesh -> None]"
      )))
    };
    let (coarse, fine) = (count(0), count(1));
    assert!(fine >= 3 * coarse, "{coarse} vs {fine}");
  }
}

/// `PlotRangePadding`: Wolfram's 3D plots pad an automatic range by
/// `Scaled[0.02]` (`ParametricPlot3D` by `Scaled[0.05]`), measured as a
/// fraction of the *padded* width, and `Scaled` coordinates are fractions
/// of that padded box.
mod range_padding {
  use super::*;

  /// The `<polygon>` vertices of an SVG, as `(x, y)` pairs.
  fn polygon_points(svg: &str) -> Vec<(f64, f64)> {
    svg
      .split("<polygon points=\"")
      .skip(1)
      .flat_map(|s| {
        s.split('"')
          .next()
          .unwrap()
          .split_whitespace()
          .filter_map(|p| {
            let (x, y) = p.split_once(',')?;
            Some((x.parse().ok()?, y.parse().ok()?))
          })
          .collect::<Vec<_>>()
      })
      .collect()
  }

  /// The end points of an SVG's `<line>` elements.
  fn line_ends(svg: &str) -> Vec<(f64, f64)> {
    let attr = |tag: &str, name: &str| -> Option<f64> {
      tag
        .split(&format!(" {name}=\""))
        .nth(1)?
        .split('"')
        .next()?
        .parse()
        .ok()
    };
    svg
      .split("<line")
      .skip(1)
      .flat_map(|s| {
        let tag = s.split('>').next().unwrap();
        [
          attr(tag, "x1").zip(attr(tag, "y1")),
          attr(tag, "x2").zip(attr(tag, "y2")),
        ]
      })
      .flatten()
      .collect()
  }

  #[test]
  fn plot3d_spells_out_its_padding() {
    clear_state();
    assert_eq!(
      interpret(
        "PlotRangePadding /. Options[Plot3D[x y, {x, 0, 1}, {y, 0, 1}, \
         PlotPoints -> 3]]"
      )
      .unwrap(),
      "{Scaled[0.02], Scaled[0.02], Scaled[0.02]}"
    );
    // An explicit vertical range is not padded.
    assert_eq!(
      interpret(
        "PlotRangePadding /. Options[Plot3D[x y, {x, 0, 1}, {y, 0, 1}, \
         PlotRange -> {0, 2}, PlotPoints -> 3]]"
      )
      .unwrap(),
      "{Scaled[0.02], Scaled[0.02], Automatic}"
    );
  }

  /// The box reaches past the surface, so a tick at the extreme of the
  /// data (`-1.0` for `Sin[x + y^2]`, whose minimum is just above -1) still
  /// falls on the box. Regression: the box ended at the data's extent and
  /// the tick was dropped.
  #[test]
  fn plot3d_box_keeps_the_extreme_ticks() {
    let s = svg(
      "Plot3D[Sin[x + y^2], {x, -3, 3}, {y, -2, 2}, PlotPoints -> 20, \
       MaxRecursion -> 1]",
    );
    let labels: Vec<&str> = s
      .split("</text>")
      .filter_map(|t| t.rsplit('>').next())
      .collect();
    assert!(
      labels
        .iter()
        .any(|l| l.ends_with("1.0") && !l.starts_with('1')),
      "{labels:?}"
    );
  }

  /// A flat surface at height `c` is boxed in `{c - |c|, c + |c|}`, as in
  /// wolframscript (`{0, 2}` for `c = 1`). Regression: `{c - 1/2, c + 1/2}`.
  #[test]
  fn plot3d_flat_surface_range() {
    let s = svg("Plot3D[1, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 3]");
    let labels: Vec<&str> = s
      .split("</text>")
      .filter_map(|t| t.rsplit('>').next())
      .collect();
    assert!(
      labels.contains(&"2.0") && labels.contains(&"1.5"),
      "{labels:?}"
    );
  }

  #[test]
  fn field_and_slice_plots_pad_their_explicit_range() {
    clear_state();
    for plot in [
      "SliceDensityPlot3D[x + y + z, \"CenterPlanes\", {x, -1, 1}, \
       {y, -1, 1}, {z, -1, 1}, PlotPoints -> 5]",
      "StreamPlot3D[{-y, x, 0.2}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       StreamPoints -> 4]",
      "VectorPlot3D[{-y, x, z}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
       VectorPoints -> 3]",
    ] {
      assert_eq!(
        interpret(&format!("PlotRangePadding /. Options[{plot}]")).unwrap(),
        "Scaled[0.02]",
        "{plot}"
      );
    }
  }

  #[test]
  fn parametric_plot3d_pads_its_surface_by_five_percent() {
    clear_state();
    assert_eq!(
      interpret(
        "PlotRangePadding /. Options[ParametricPlot3D[{u, v, u v}, \
         {u, 0, 1}, {v, 0, 1}, PlotPoints -> 3]]"
      )
      .unwrap(),
      "{{Scaled[0.05], Scaled[0.05]}, {Scaled[0.05], Scaled[0.05]}, \
       {Scaled[0.05], Scaled[0.05]}}"
    );
  }

  /// `Scaled[{0, 0, 0}]` … `Scaled[{1, 1, 0}]` is the floor of the padded
  /// box: its corners are the corners of the drawn box. Regression: they
  /// were fractions of the unpadded range, inside the box.
  #[test]
  fn scaled_points_are_fractions_of_the_padded_box() {
    let s = svg(
      "Graphics3D[{Cuboid[], Polygon[{Scaled[{0, 0, 0}], \
       Scaled[{1, 0, 0}], Scaled[{1, 1, 0}], Scaled[{0, 1, 0}]}]}, \
       Axes -> False]",
    );
    let corners = line_ends(&s);
    let on_box_corner = |p: &(f64, f64)| {
      corners
        .iter()
        .any(|c| (c.0 - p.0).abs() < 0.2 && (c.1 - p.1).abs() < 0.2)
    };
    let mut hits: Vec<(i64, i64)> = polygon_points(&s)
      .iter()
      .filter(|p| on_box_corner(p))
      .map(|p| (p.0.round() as i64, p.1.round() as i64))
      .collect();
    hits.sort_unstable();
    hits.dedup();
    assert!(hits.len() >= 4, "{hits:?}\n{s}");
  }

  /// A 3D bar chart's invisible base slab is 0.024 high whatever the
  /// values. Regression: it grew with the tallest bar.
  #[test]
  fn bar_chart3d_base_slab_is_fixed_height() {
    clear_state();
    assert_eq!(
      interpret(
        "Cases[BarChart3D[{10, 20}], Cuboid[{_, _, z0_}, {_, _, z1_}] :> \
         {z0, z1}, Infinity]"
      )
      .unwrap(),
      "{{0, 0.024}}"
    );
    assert_eq!(
      interpret(
        "Cases[BarChart3D[{-5, 2}], Cuboid[{_, _, z0_}, {_, _, z1_}] :> \
         {z0, z1}, Infinity]"
      )
      .unwrap(),
      "{{-0.02, 0.024}}"
    );
  }

  /// `RectangleChart3D` stands its blocks on the same backdrop as
  /// `BarChart3D`: a beveled floor and back wall in `Scaled` coordinates,
  /// over the invisible base slab. Regression: it had neither.
  #[test]
  fn rectangle_chart3d_has_the_chart_backdrop() {
    clear_state();
    assert_eq!(
      interpret(
        "g = RectangleChart3D[{{1, 2, 3}, {2, 1, 2}, {3, 3, 1}}]; \
         {Length[Cases[g, Polygon[{__Scaled}, ___], Infinity]], \
         Cases[g, Cuboid[a_, b_] :> {a, b}, Infinity]}"
      )
      .unwrap(),
      "{12, {{{-0.30970000000000003, -0.75, 0}, {6.5037, 3.75, 0.024}}}}"
    );
  }

  /// `VectorPlot3D` sizes its heads as Wolfram does: `Arrowheads` of `0.3`
  /// times the longest arrow (as a fraction of the box side) and tubes a
  /// twentieth of that arrow thick, drawn by the `Arrow[Tube[…]]` renderer.
  #[test]
  fn vector_plot3d_arrowheads_follow_the_longest_arrow() {
    clear_state();
    assert_eq!(
      interpret(
        "g = VectorPlot3D[{-y, x, z}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, \
         VectorPoints -> 4]; \
         {Max[Abs[Cases[g, Arrowheads[{{s_, _}}, ___] :> s, Infinity] - \
         0.0525]] < 10^-9, Max[Abs[Cases[g, Tube[_, r_] :> r, Infinity] - \
         0.0175]] < 10^-9, Count[g, CapForm[\"Butt\"], Infinity]}"
      )
      .unwrap(),
      "{True, True, 1}"
    );
  }

  /// A `Graphics3D` `Text` label whose inline box notation hangs limits on
  /// a large operator is drawn as the same laid-out picture the 2D `Text`
  /// uses (a tall sign with its limits at its ends), in the label's color
  /// and size, while other labels of the scene stay plain text.
  #[test]
  fn text3d_large_operator_limits_are_laid_out() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Graphics3D[{Red, Text[Row[{"volume = ", "\!\(\*SubsuperscriptBox[\(\[Integral]\), \(0\), \(h\)]\)", Style["A", Italic], Style[" dy", Italic]}], {0, 0, 1}], Text[Row[{"n = ", "\!\(\*SubscriptBox[\(x\), \(1\)]\)"}], {0, 0, 0}]}, BaseStyle -> 20], "SVG"]"#,
    )
    .unwrap();
    let at = svg.find("<svg x=").expect("embedded label picture");
    let picture = &svg[at..at + svg[at..].find("</svg>").unwrap()];
    assert!(
      picture.contains("<path d=\"M "),
      "no sign stroke: {picture}"
    );
    assert!(picture.contains("rgb(255,0,0)"), "{picture}");
    assert!(picture.contains("font-size=\"20.0\""), "{picture}");
    for part in [">volume = </text>", ">h</text>", ">0</text>", "> dy</text>"] {
      assert!(picture.contains(part), "{part} missing in {picture}");
    }
    assert_eq!(svg.matches("<svg x=").count(), 1, "{svg}");
    assert!(
      svg.contains(">n = <tspan font-style=\"italic\">x</tspan><tspan"),
      "{svg}"
    );
  }
}

/// A `Point` in a `Graphics3D` without a `PointSize` takes Wolfram's 3D
/// default, `PointSize[0.01]` of the picture's width.
mod default_point_size {
  use super::*;

  fn first_radius(svg: &str) -> f64 {
    svg
      .split("<circle")
      .nth(1)
      .and_then(|s| s.split(" r=\"").nth(1))
      .and_then(|s| s.split('"').next())
      .and_then(|r| r.parse().ok())
      .unwrap_or_else(|| panic!("no circle in {svg}"))
  }

  #[test]
  fn scales_with_the_image_width() {
    for width in [200.0, 360.0, 600.0] {
      let s = svg(&format!(
        "Graphics3D[Point[{{{{0, 0, 0}}, {{1, 1, 1}}}}], ImageSize -> {width}]"
      ));
      let r = first_radius(&s);
      assert!((r - 0.005 * width).abs() < 0.02, "{width}: r = {r}");
    }
  }

  #[test]
  fn uses_the_shorter_side_of_a_wide_pinned_picture() {
    let s = svg(
      "Graphics3D[{Red, Point[{{0, 0, 0}, {1, 1, 1}}]}, \
       ImageSize -> {500, 400}]",
    );
    assert!((first_radius(&s) - 2.0).abs() < 0.02, "{s}");
  }

  #[test]
  fn an_explicit_point_size_still_wins() {
    let s = svg(
      "Graphics3D[{PointSize[0.03], Point[{{0, 0, 0}, {1, 1, 1}}]}, \
       ImageSize -> 400]",
    );
    assert!((first_radius(&s) - 6.0).abs() < 0.02, "{s}");
  }
}

/// `ParametricPlot3D` records its `PlotRange` and `PlotRangePadding` axis
/// by axis, as wolframscript does: an automatic axis resolved to the
/// surface's extent and padded by `Scaled[0.05]`, an `All` one kept as
/// `All` and padded, an explicit one as given and unpadded. A `Show` led by
/// the plot frames that box.
mod parametric_plot_range_all {
  use super::*;

  const SCALED: &str = "{Scaled[0.05], Scaled[0.05]}";

  fn recorded(options: &str) -> String {
    interpret(&format!(
      "Options[ParametricPlot3D[{{s, t, 2 s t}}, {{s, 0, 1}}, {{t, 0, 1}}\
       {options}], {{PlotRange, PlotRangePadding}}]"
    ))
    .unwrap()
  }

  #[test]
  fn resolves_the_range_and_records_the_padding() {
    clear_state();
    for spec in ["", ", PlotRange -> Automatic"] {
      assert_eq!(
        recorded(spec),
        format!(
          "{{PlotRange -> {{{{0., 1.}}, {{0., 1.}}, {{0., 2.}}}}, \
           PlotRangePadding -> {{{SCALED}, {SCALED}, {SCALED}}}}}"
        )
      );
    }
    assert_eq!(
      recorded(", PlotRange -> All"),
      format!(
        "{{PlotRange -> {{All, All, All}}, \
         PlotRangePadding -> {{{SCALED}, {SCALED}, {SCALED}}}}}"
      )
    );
  }

  #[test]
  fn an_explicit_range_stays_unpadded() {
    clear_state();
    assert_eq!(
      interpret(
        "Options[ParametricPlot3D[{s, t, s t}, {s, 0, 1}, {t, 0, 1}, \
         PlotRange -> {{0, 2}, {0, 2}, {0, 2}}], \
         {PlotRange, PlotRangePadding}]"
      )
      .unwrap(),
      "{PlotRange -> {{0, 2}, {0, 2}, {0, 2}}, \
       PlotRangePadding -> {{0, 0}, {0, 0}, {0, 0}}}"
    );
  }

  #[test]
  fn each_axis_is_padded_by_its_own_kind() {
    clear_state();
    assert_eq!(
      recorded(", PlotRange -> {All, {0, 1}, Automatic}"),
      format!(
        "{{PlotRange -> {{All, {{0, 1}}, {{0., 2.}}}}, \
         PlotRangePadding -> {{{SCALED}, {{0, 0}}, {SCALED}}}}}"
      )
    );
    // A bare number is a symmetric range; a single pair is the vertical
    // range, recorded for the other axes too, which stay padded; two pairs
    // are the horizontal ranges.
    assert_eq!(
      recorded(", PlotRange -> 3"),
      "{PlotRange -> {{-3, 3}, {-3, 3}, {-3, 3}}, \
       PlotRangePadding -> {{0, 0}, {0, 0}, {0, 0}}}"
    );
    assert_eq!(
      recorded(", PlotRange -> {0, 1/2}"),
      format!(
        "{{PlotRange -> {{{{0, 1/2}}, {{0, 1/2}}, {{0, 1/2}}}}, \
         PlotRangePadding -> {{{SCALED}, {SCALED}, {{0, 0}}}}}}"
      )
    );
    assert_eq!(
      recorded(", PlotRange -> {{0, 1/2}, {0, 1/2}}"),
      format!(
        "{{PlotRange -> {{{{0, 1/2}}, {{0, 1/2}}, {{0., 2.}}}}, \
         PlotRangePadding -> {{{{0, 0}}, {{0, 0}}, {SCALED}}}}}"
      )
    );
  }

  #[test]
  fn an_explicit_padding_is_spelled_out_per_side() {
    clear_state();
    assert_eq!(
      recorded(", PlotRangePadding -> {1, {1, 2}, Automatic}"),
      format!(
        "{{PlotRange -> {{{{0., 1.}}, {{0., 1.}}, {{0., 2.}}}}, \
         PlotRangePadding -> {{{{1, 1}}, {{1, 2}}, {SCALED}}}}}"
      )
    );
    assert_eq!(
      recorded(
        ", PlotRange -> {{0, 2}, {0, 2}, {0, 2}}, PlotRangePadding -> None"
      ),
      "{PlotRange -> {{0, 2}, {0, 2}, {0, 2}}, \
       PlotRangePadding -> {{None, None}, {None, None}, {None, None}}}"
    );
  }

  #[test]
  fn a_space_curve_records_its_range_too() {
    clear_state();
    assert_eq!(
      interpret(
        "Options[ParametricPlot3D[{t, t^2, 2 t}, {t, 0, 1}, \
         PlotRange -> {{0, 2}, All, {0, 3}}], {PlotRange, PlotRangePadding}]"
      )
      .unwrap(),
      "{PlotRange -> {{0, 2}, All, {0, 3}}, PlotRangePadding -> \
       {{0, 0}, {Scaled[0.05], Scaled[0.05]}, {0, 0}}}"
    );
  }

  #[test]
  fn a_show_led_by_it_frames_the_padded_box() {
    let show = |pp_range: &str| {
      svg(&format!(
        "Show[ParametricPlot3D[{{s, t, s t}}, {{s, 0, 1}}, {{t, 0, 1}}, \
         Mesh -> None, PlotRange -> {pp_range}], \
         PlotRange -> {{{{-1, 2}}, {{-1, 2}}, {{-1, 2}}}}, ViewAngle -> 0.7, \
         Boxed -> False, Axes -> None, ImageSize -> 400]"
      ))
    };
    // The padded box is 1.1 times as wide, so the scene is drawn smaller
    // than with the unpadded explicit range, and like the unnamed range.
    assert_eq!(show("All"), show("Automatic"));
    assert_ne!(
      show("All"),
      show("{{0, 1}, {0, 1}, {0, 1}}"),
      "PlotRange -> All must pad like an automatic range"
    );
  }
}
