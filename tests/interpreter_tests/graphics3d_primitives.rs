use super::*;

fn svg(expr: &str) -> String {
  clear_state();
  interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
}

/// How many filled `<polygon>` elements an SVG draws.
fn polygon_count(svg: &str) -> usize {
  svg.matches("<polygon").count()
}

/// The distinct `fill="rgb(…)"` colours an SVG uses.
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

  // The corner colours are blended across the face instead of the face
  // being drawn in one flat colour.
  #[test]
  fn polygon_vertex_colors_blend_across_the_face() {
    let s = svg(
      "Graphics3D[Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 1}, {0, 1, 1}}, \
       VertexColors -> {Red, Green, Blue, Yellow}]]",
    );
    let colours = fills(&s);
    assert!(colours.len() > 50, "got: {colours:?}");
    // Near the red corner the face is (lit) red: no green or blue.
    assert!(
      colours.iter().any(|c| {
        let v: Vec<u8> =
          c.split(',').map(|x| x.trim().parse().unwrap()).collect();
        v[0] > 150 && v[1] < 30 && v[2] < 30
      }),
      "got: {colours:?}"
    );
  }

  #[test]
  fn line_vertex_colors_colour_each_segment() {
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
  fn draws_coloured_streamlines() {
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
}

mod outlines {
  use super::*;

  // Cylinder and cone rims are outlined by the default edge, like
  // wolframscript's; `EdgeForm[]` removes them.
  #[test]
  fn rims_are_outlined_by_default() {
    let lines = |s: &str| s.lines().filter(|l| l.starts_with("<line")).count();
    let plain = svg("Graphics3D[Cylinder[], Boxed -> False]");
    let bare = svg("Graphics3D[{EdgeForm[], Cylinder[]}, Boxed -> False]");
    assert!(lines(&plain) > 10, "{plain}");
    assert_eq!(lines(&bare), 0);
    let cone = svg("Graphics3D[Cone[], Boxed -> False]");
    assert!(lines(&cone) > 10, "{cone}");
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
      .filter(|l| l.starts_with("<line") && l.contains(stroke))
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
    let colours = fills(&s);
    let reddish = colours.iter().any(|c| {
      let v: Vec<u8> =
        c.split(',').map(|x| x.trim().parse().unwrap()).collect();
      v[0] > 150 && v[1] < 60
    });
    let yellowish = colours.iter().any(|c| {
      let v: Vec<u8> =
        c.split(',').map(|x| x.trim().parse().unwrap()).collect();
      v[0] > 150 && v[1] > 150 && v[2] < 60
    });
    assert!(reddish && yellowish, "{colours:?}");
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
  // default colour (the first plot colour, not black).
  #[test]
  fn thick_space_curve_keeps_its_colour() {
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

  // Each contour level takes the next theme colour (orange, blue, green),
  // as in wolframscript.
  #[test]
  fn contour_levels_take_theme_colours() {
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
        let v: Vec<u8> =
          c.split(',').map(|x| x.trim().parse().unwrap()).collect();
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
