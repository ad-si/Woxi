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
