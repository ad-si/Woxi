use super::*;

/// The legacy `PolyhedronOperations` package's `Truncate` and `Stellate`,
/// implemented under their fully-qualified names since Woxi has no package
/// system for `Needs["PolyhedronOperations`"]` to load into.
mod polyhedron_operations_tests {
  use super::*;

  #[test]
  fn truncate_corner_cuts_a_square() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Truncate[Polygon[{{0,0,0},{2,0,0},{2,2,0},{0,2,0}}], 1/4]"
      )
      .unwrap(),
      "Polygon[{{0.5, 0., 0.}, {1.5, 0., 0.}, {2., 0.5, 0.}, {2., 1.5, 0.}, \
       {1.5, 2., 0.}, {0.5, 2., 0.}, {0., 1.5, 0.}, {0., 0.5, 0.}}]"
    );
  }

  #[test]
  fn truncate_default_ratio_is_three_tenths() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Truncate[Polygon[{{0,0,0},{2,0,0},{2,2,0},{0,2,0}}]]"
      )
      .unwrap(),
      "Polygon[{{0.6, 0., 0.}, {1.4, 0., 0.}, {2., 0.6, 0.}, {2., 1.4, 0.}, \
       {1.4, 2., 0.}, {0.6000000000000001, 2., 0.}, {0., 1.4, 0.}, \
       {0., 0.6000000000000001, 0.}}]"
    );
  }

  #[test]
  fn truncate_ratio_outside_zero_to_half_stays_unevaluated() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Truncate[Polygon[{{0,0,0},{1,0,0},{1,1,0}}], 0.6]"
      )
      .unwrap(),
      "PolyhedronOperations`Truncate[Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}], 0.6]"
    );
  }

  #[test]
  fn truncate_symbolic_ratio_stays_unevaluated() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Truncate[Polygon[{{0,0,0},{1,0,0},{1,1,0}}], x]"
      )
      .unwrap(),
      "PolyhedronOperations`Truncate[Polygon[{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}], x]"
    );
  }

  #[test]
  fn truncate_recurses_into_a_list_of_faces() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Truncate[{Polygon[{{0,0,0},{2,0,0},{2,2,0},{0,2,0}}]}, 1/4]"
      )
      .unwrap(),
      "{Polygon[{{0.5, 0., 0.}, {1.5, 0., 0.}, {2., 0.5, 0.}, {2., 1.5, 0.}, \
       {1.5, 2., 0.}, {0.5, 2., 0.}, {0., 1.5, 0.}, {0., 0.5, 0.}}]}"
    );
  }

  #[test]
  fn stellate_replaces_a_triangle_with_three_pyramid_faces() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Stellate[Polygon[{{1,0,0},{0,1,0},{0,0,1}}], 2]"
      )
      .unwrap(),
      "{Polygon[{{1., 0., 0.}, {0., 1., 0.}, \
       {0.6666666666666666, 0.6666666666666666, 0.6666666666666666}}], \
       Polygon[{{0., 1., 0.}, {0., 0., 1.}, \
       {0.6666666666666666, 0.6666666666666666, 0.6666666666666666}}], \
       Polygon[{{0., 0., 1.}, {1., 0., 0.}, \
       {0.6666666666666666, 0.6666666666666666, 0.6666666666666666}}]}"
    );
  }

  #[test]
  fn stellate_default_ratio_is_two() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Stellate[Polygon[{{1,0,0},{0,1,0},{0,0,1}}]]"
      )
      .unwrap(),
      interpret(
        "PolyhedronOperations`Stellate[Polygon[{{1,0,0},{0,1,0},{0,0,1}}], 2]"
      )
      .unwrap()
    );
  }

  /// A stellation ratio of 1 puts the apex exactly at the face centroid —
  /// the pyramid degenerates to a flat fan of triangles at the original
  /// face's own distance from the origin, as documented ("ratios less
  /// than 1 give concave figures", implying ratio 1 is the flat boundary).
  #[test]
  fn stellate_ratio_one_places_apex_at_centroid() {
    assert_eq!(
      interpret(
        "PolyhedronOperations`Stellate[Polygon[{{1,0,0},{0,1,0},{0,0,1}}], 1]"
      )
      .unwrap(),
      "{Polygon[{{1., 0., 0.}, {0., 1., 0.}, \
       {0.3333333333333333, 0.3333333333333333, 0.3333333333333333}}], \
       Polygon[{{0., 1., 0.}, {0., 0., 1.}, \
       {0.3333333333333333, 0.3333333333333333, 0.3333333333333333}}], \
       Polygon[{{0., 0., 1.}, {1., 0., 0.}, \
       {0.3333333333333333, 0.3333333333333333, 0.3333333333333333}}]}"
    );
  }

  /// `Rotate[g, θ, axis]` stays a symbolic wrapper around `g` outside a
  /// rendered graphic, so `Truncate`/`Stellate` must resolve it to
  /// concrete coordinates before finding the `Polygon` faces inside —
  /// otherwise a rotated face would pass through untouched.
  #[test]
  fn truncate_resolves_a_rotate_wrapped_face() {
    // Compare the two point sets by their (chopped) difference rather than
    // as strings: `Sin[Pi/2]`/`Cos[Pi/2]` carry ~1 ulp of floating-point
    // rotation error, which `Chop` only clears from a value near zero, not
    // from a coordinate near -1 that is a few ulps off exact — but the
    // *difference* between the two computations is near zero regardless.
    let code = "\
      direct = PolyhedronOperations`Truncate[ \
        Polygon[{{0,1,0},{0,2,0},{-1,2,0},{-1,1,0}}], 1/4][[1]]; \
      rotated = PolyhedronOperations`Truncate[Rotate[ \
        Polygon[{{1,0,0},{2,0,0},{2,1,0},{1,1,0}}], Pi/2, {0,0,1}], 1/4][[1]]; \
      Union[Flatten[Chop[direct - rotated]]]";
    assert_eq!(interpret(code).unwrap(), "{0}");
  }

  /// `Needs["PolyhedronOperations`"]` has nothing to load (Woxi has no
  /// package system, see `evaluator::contexts`) but must not error, and
  /// the fully-qualified functions must keep working after it runs.
  #[test]
  fn needs_polyhedron_operations_is_a_harmless_no_op() {
    assert_eq!(
      interpret(
        r#"Needs["PolyhedronOperations`"]; PolyhedronOperations`Truncate[Polygon[{{0,0,0},{2,0,0},{2,2,0},{0,2,0}}]]"#
      )
      .unwrap(),
      "Polygon[{{0.6, 0., 0.}, {1.4, 0., 0.}, {2., 0.6, 0.}, {2., 1.4, 0.}, \
       {1.4, 2., 0.}, {0.6000000000000001, 2., 0.}, {0., 1.4, 0.}, \
       {0., 0.6000000000000001, 0.}}]"
    );
  }

  /// End-to-end: a ring of bands (its own construction, not derived from
  /// any Demonstration) truncated then stellated, wrapped in `Graphics3D`
  /// and exported — every face must still be a Polygon and the export
  /// must produce non-empty SVG.
  #[test]
  fn truncate_then_stellate_renders_as_graphics3d() {
    let code = "\
      n = 8; m = 6; \
      band = Table[Polygon[{ \
          {Cos[t], Sin[t], -0.5}, \
          {Cos[t + 2 Pi/n], Sin[t + 2 Pi/n], -0.5}, \
          {Cos[t + 2 Pi/n], Sin[t + 2 Pi/n], 0.5}, \
          {Cos[t], Sin[t], 0.5} \
        }], {t, 0, 2 Pi - 2 Pi/n, 2 Pi/n}]; \
      ring = Table[Rotate[band, i (2 Pi/m), {0, 1, 0}], {i, m}]; \
      stellated = PolyhedronOperations`Stellate[ \
        PolyhedronOperations`Truncate[ring, 0.3], 1.7]; \
      svg = ExportString[Graphics3D[{Yellow, stellated}, Boxed -> False], \"SVG\"]; \
      {Length[Flatten[{stellated}]], StringLength[svg] > 0}";
    // n faces × m copies, each an 8-gon after truncation (n=8) → n triangles
    // per stellated face: n * m * n = 8*6*8 = 384.
    assert_eq!(interpret(code).unwrap(), "{384, True}");
  }

  const CUBE: &str = "GraphicsComplex[\
    {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}}, \
    Polygon[{{1,4,3,2},{5,6,7,8},{1,2,6,5},{2,3,7,6},{3,4,8,7},{4,1,5,8}}]]";

  /// A `GraphicsComplex` solid (what `PolyhedronData` coordinates and face
  /// indices build) is truncated face by face, and each of its corners is
  /// capped by a new face through the cut points around it — a cube turns
  /// into 6 octagons and 8 triangles.
  #[test]
  fn truncate_graphics_complex_caps_every_corner() {
    let code = format!(
      "t = PolyhedronOperations`Truncate[{CUBE}, 1/4]; \
       Sort[Length[First[#]] & /@ Cases[t, _Polygon, Infinity]]"
    );
    assert_eq!(
      interpret(&code).unwrap(),
      "{3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8}"
    );
  }

  /// The corner cap at the origin passes through the three cut points next
  /// to it and faces outward, like the solid's own faces.
  #[test]
  fn truncate_corner_cap_points_and_orientation() {
    let code = format!(
      "c = First[First[Cases[PolyhedronOperations`Truncate[{CUBE}, 1/4], \
         Polygon[p_ /; Length[p] == 3 && Total[Flatten[p]] < 1], Infinity]]]; \
       {{Sort[c], Sign[Cross[c[[2]] - c[[1]], c[[3]] - c[[1]]] . (c[[1]] - {{1, 1, 1}}/2)]}}"
    );
    assert_eq!(
      interpret(&code).unwrap(),
      "{{{0., 0., 0.25}, {0., 0.25, 0.}, {0.25, 0., 0.}}, 1}"
    );
  }

  /// A ratio of 0 cuts nothing off: every face keeps its corners (each one
  /// doubled) and no cap is added.
  #[test]
  fn truncate_ratio_zero_keeps_the_solid() {
    let code = format!(
      "t = PolyhedronOperations`Truncate[{CUBE}, 0]; \
       {{Length[Cases[t, _Polygon, Infinity]], \
         Union[Flatten[Map[Union, Cases[t, Polygon[p_] :> p, Infinity]], 1]] == \
         Tuples[{{0., 1.}}, 3]}}"
    );
    assert_eq!(interpret(&code).unwrap(), "{6, True}");
  }

  /// Loading the package puts its context on `$ContextPath`, so a bare
  /// `Truncate` read afterwards is the package's function.
  #[test]
  fn bare_truncate_resolves_after_loading_the_package() {
    for load in ["Needs", "Get"] {
      assert_eq!(
        interpret(&format!(
          "{load}[\"PolyhedronOperations`\"]\n\
           {{Context[Truncate], Context[Stellate], \
             Truncate[Polygon[{{{{0,0,0}},{{2,0,0}},{{2,2,0}},{{0,2,0}}}}], 1/4]}}"
        ))
        .unwrap(),
        "{PolyhedronOperations`, PolyhedronOperations`, \
         Polygon[{{0.5, 0., 0.}, {1.5, 0., 0.}, {2., 0.5, 0.}, {2., 1.5, 0.}, \
         {1.5, 2., 0.}, {0.5, 2., 0.}, {0., 1.5, 0.}, {0., 0.5, 0.}}]}"
      );
    }
  }
}
