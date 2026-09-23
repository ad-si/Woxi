use super::*;

mod knot_data_tests {
  use super::*;

  // The Trefoil is the (2, 3)-torus knot: 3 crossings, Alexander-Briggs
  // notation 3_1.
  #[test]
  fn knot_data_trefoil_basics() {
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "CrossingNumber"]"#).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "AlexanderBriggsNotation"]"#).unwrap(),
      "3_1"
    );
  }

  // Aliases resolve to the same knot.
  #[test]
  fn knot_data_aliases() {
    for name in ["Trefoil", "TrefoilKnot", "3_1"] {
      assert_eq!(
        interpret(&format!(r#"KnotData["{name}", "CrossingNumber"]"#)).unwrap(),
        "3",
        "name: {name}"
      );
    }
  }

  // KnotData[name, "SpaceCurve"] is a Function[{t}, {x, y, z}] that can be
  // applied to a variable and then evaluated numerically.
  #[test]
  fn knot_data_space_curve_is_a_function_of_t() {
    assert_eq!(
      interpret(r#"Head[KnotData["Trefoil", "SpaceCurve"]]"#).unwrap(),
      "Function"
    );
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "SpaceCurve"][0]"#).unwrap(),
      "{3, 0, 0}"
    );
    // The curve closes up after one period.
    assert_eq!(
      interpret(
        r#"Chop[KnotData["Trefoil", "SpaceCurve"][2 Pi] -
             KnotData["Trefoil", "SpaceCurve"][0]] // N"#
      )
      .unwrap(),
      "{0., 0., 0.}"
    );
  }

  // Every point of the space curve sits at distance 1 from the core torus
  // circle of radius 2, since it winds around a unit tube.
  #[test]
  fn knot_data_space_curve_lies_on_unit_tube() {
    let result = interpret(
      r#"With[{r = KnotData["Trefoil", "SpaceCurve"]},
           Table[
             Round[(Sqrt[r[t][[1]]^2 + r[t][[2]]^2] - 2)^2 + r[t][[3]]^2, 10^-9],
             {t, 0, 2 Pi, Pi/5}] // N]"#,
    )
    .unwrap();
    assert_eq!(
      result, "{1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}",
      "got: {result}"
    );
  }

  // The general `{"TorusKnot", {p, q}}` family covers knots beyond the
  // handful of named entries, and its crossing number follows the same
  // min(p(q-1), q(p-1)) formula.
  #[test]
  fn knot_data_general_torus_knot() {
    assert_eq!(
      interpret(r#"KnotData[{"TorusKnot", {2, 5}}, "CrossingNumber"]"#)
        .unwrap(),
      "5"
    );
    assert_eq!(
      interpret(r#"KnotData[{"TorusKnot", {3, 5}}, "CrossingNumber"]"#)
        .unwrap(),
      "10"
    );
    // Named entries and their explicit torus-knot spec agree.
    assert_eq!(
      interpret(
        r#"KnotData["CinquefoilKnot", "CrossingNumber"] ==
             KnotData[{"TorusKnot", {2, 5}}, "CrossingNumber"]"#
      )
      .unwrap(),
      "True"
    );
  }

  // A non-coprime {p, q} pair does not describe a knot (it is a link, or
  // degenerates), so it is rejected and the call stays unevaluated.
  #[test]
  fn knot_data_rejects_non_coprime_winding_numbers() {
    assert_eq!(
      interpret(r#"KnotData[{"TorusKnot", {2, 4}}, "CrossingNumber"]"#)
        .unwrap(),
      r#"KnotData[{TorusKnot, {2, 4}}, CrossingNumber]"#
    );
  }

  // Unknown entities stay unevaluated (with a notent message).
  #[test]
  fn knot_data_unknown_name() {
    assert_eq!(
      interpret(r#"KnotData["NoSuchKnot", "CrossingNumber"]"#).unwrap(),
      r#"KnotData[NoSuchKnot, CrossingNumber]"#
    );
  }

  // Unknown properties stay unevaluated.
  #[test]
  fn knot_data_unknown_property() {
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "NoSuchProperty"]"#).unwrap(),
      "KnotData[Trefoil, NoSuchProperty]"
    );
  }

  // KnotData["Properties"] lists the supported property names.
  #[test]
  fn knot_data_properties() {
    assert_eq!(
      interpret(r#"KnotData["Properties"]"#).unwrap(),
      "{AlexanderBriggsNotation, CrossingNumber, ImageData, SpaceCurve}"
    );
  }

  // KnotData[All] lists the named entities.
  #[test]
  fn knot_data_all_lists_named_entities() {
    assert_eq!(
      interpret("KnotData[All]").unwrap(),
      "{CinquefoilKnot, SeptafoilKnot, Trefoil}"
    );
  }

  // KnotData[name] renders the knot as a 3D parametric plot.
  #[test]
  fn knot_data_renders_graphics3d() {
    assert_eq!(interpret(r#"KnotData["Trefoil"]"#).unwrap(), "-Graphics3D-");
  }

  // KnotData[name, "ImageData"] is a list holding a single GraphicsComplex
  // mesh (points + Polygon faces) that can be used as a Graphics3D
  // primitive directly.
  #[test]
  fn knot_data_image_data_is_a_graphics_complex() {
    assert_eq!(
      interpret(r#"Head[KnotData["Trefoil", "ImageData"]]"#).unwrap(),
      "List"
    );
    assert_eq!(
      interpret(r#"Length[KnotData["Trefoil", "ImageData"]]"#).unwrap(),
      "1"
    );
    assert_eq!(
      interpret(r#"Head[First[KnotData["Trefoil", "ImageData"]]]"#).unwrap(),
      "GraphicsComplex"
    );
  }

  // The mesh is a tube swept around the space curve: each ring of points
  // around one cross-section is centered exactly on that point of the
  // curve (the ring's offsets from center are evenly spaced around a
  // circle, so they cancel out in the average).
  #[test]
  fn knot_data_image_data_rings_are_centered_on_the_space_curve() {
    let result = interpret(
      r#"With[{r = KnotData["Trefoil", "SpaceCurve"],
              pts = First[KnotData["Trefoil", "ImageData"]][[1]]},
           With[{centers = Mean /@ Partition[pts, Length[pts]/96]},
             Max[Norm /@ (centers -
               Table[r[2 Pi k/96], {k, 0, 95}])] < 10^-9]]"#,
    )
    .unwrap();
    assert_eq!(result, "True", "got: {result}");
  }

  // The mesh renders fine as a Graphics3D primitive, including under
  // Scale[…] as used by demonstrations that place several copies of the
  // knot around a circle.
  #[test]
  fn knot_data_image_data_renders_in_graphics3d() {
    assert_eq!(
      interpret(
        r#"Head[Graphics3D[Scale[KnotData[{"TorusKnot", {2, 7}}, "ImageData"], 6]]]"#
      )
      .unwrap(),
      "Graphics3D"
    );
  }

  // Unlike SpaceCurve/CrossingNumber, ImageData is defined for every
  // coprime {p, q}, not only the three named entries.
  #[test]
  fn knot_data_image_data_general_torus_knot() {
    assert_eq!(
      interpret(r#"Head[First[KnotData[{"TorusKnot", {3, 4}}, "ImageData"]]]"#)
        .unwrap(),
      "GraphicsComplex"
    );
  }
}
