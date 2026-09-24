use super::*;

mod knot_data_tests {
  use super::*;

  // The Trefoil is the first 3-crossing knot of the Rolfsen table, 3_1.
  #[test]
  fn knot_data_trefoil_basics() {
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "CrossingNumber"]"#).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "AlexanderBriggsNotation"]"#).unwrap(),
      "Subscript[3, 1]"
    );
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "AlexanderBriggsList"]"#).unwrap(),
      "{3, 1}"
    );
  }

  // A knot is named by its table entry {n, k} or by its standard name.
  #[test]
  fn knot_data_table_entries_and_names() {
    assert_eq!(
      interpret(
        r#"Table[{KnotData[k, "StandardName"], KnotData[k, "Name"],
             KnotData[k, "CrossingNumber"]},
           {k, {{0, 1}, {3, 1}, {4, 1}, {5, 1}, {5, 2}, {6, 1}, {10, 161}}}]"#
      )
      .unwrap(),
      "{{Unknot, unknot, 0}, {Trefoil, trefoil, 3}, \
       {FigureEight, figure eight knot, 4}, \
       {SolomonSeal, Solomon seal knot, 5}, {{Knot, {5, 2}}, knot 5-2, 5}, \
       {Stevedore, Stevedore knot, 6}, {PerkoPair, Perko pair, 10}}"
    );
    assert_eq!(
      interpret("KnotData[]").unwrap(),
      "{Unknot, Trefoil, FigureEight, SolomonSeal, Stevedore, PerkoPair}"
    );
  }

  // KnotData[All] lists the Rolfsen table: every prime knot of up to ten
  // crossings, plus the unknot.
  #[test]
  fn knot_data_all_is_the_knot_table() {
    assert_eq!(
      interpret(
        "{Length[KnotData[All]], Take[KnotData[All], 5], \
         Last[KnotData[All]], Counts[First /@ KnotData[All]]}"
      )
      .unwrap(),
      "{250, {{0, 1}, {3, 1}, {4, 1}, {5, 1}, {5, 2}}, {10, 165}, \
       <|0 -> 1, 3 -> 1, 4 -> 1, 5 -> 2, 6 -> 3, 7 -> 7, 8 -> 21, 9 -> 49, \
       10 -> 165|>}"
    );
  }

  // KnotData[name, "SpaceCurve"] is a pure function of the curve
  // parameter, the trefoil's the classic
  // {Sin[t] + 2 Sin[2 t], Cos[t] - 2 Cos[2 t], -Sin[3 t]}.
  #[test]
  fn knot_data_space_curve_is_a_function_of_t() {
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "SpaceCurve"]"#).unwrap(),
      "{Sin[#1] + 2*Sin[2*#1], Cos[#1] - 2*Cos[2*#1], -Sin[3*#1]} & "
    );
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "SpaceCurve"][0]"#).unwrap(),
      "{0, -1, 0}"
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

  // A torus knot's space curve winds around a unit tube about the core
  // circle of radius 2: every point is at distance 1 from it.
  #[test]
  fn knot_data_torus_space_curve_lies_on_unit_tube() {
    assert_eq!(
      interpret(r#"KnotData[{"TorusKnot", {2, 5}}, "SpaceCurve"]"#).unwrap(),
      "{(2 + Cos[5*#1])*Cos[2*#1], (2 + Cos[5*#1])*Sin[2*#1], Sin[5*#1]} & "
    );
    let result = interpret(
      r#"With[{r = KnotData[{"TorusKnot", {2, 5}}, "SpaceCurve"]},
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

  // The general `{"TorusKnot", {p, q}}` family covers every torus knot;
  // its crossing number follows the min(p(q-1), q(p-1)) formula, and it
  // has no place in the knot table.
  #[test]
  fn knot_data_general_torus_knot() {
    assert_eq!(
      interpret(
        r#"{KnotData[{"TorusKnot", {2, 5}}, "CrossingNumber"],
            KnotData[{"TorusKnot", {3, 5}}, "CrossingNumber"],
            KnotData[{"TorusKnot", {3, 4}}, "AlexanderBriggsList"],
            KnotData[{"TorusKnot", {3, 4}}, "Name"],
            KnotData[{"TorusKnot", {3, 4}}, "StandardName"]}"#
      )
      .unwrap(),
      "{5, 10, Missing[NotApplicable], (3,4)-torus knot, {TorusKnot, {3, 4}}}"
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

  // Unknown entities stay unevaluated (with a notent message) — an
  // Alexander–Briggs label is not a name, and the table ends at ten
  // crossings.
  #[test]
  fn knot_data_unknown_name() {
    for (code, shown) in [
      (r#"KnotData["NoSuchKnot", "CrossingNumber"]"#, "NoSuchKnot"),
      (r#"KnotData["3_1", "CrossingNumber"]"#, "3_1"),
      (r#"KnotData[{11, 1}, "CrossingNumber"]"#, "{11, 1}"),
    ] {
      assert_eq!(
        interpret(code).unwrap(),
        code.replace('"', ""),
        "{code} stays unevaluated"
      );
      assert_eq!(
        woxi::get_captured_messages_raw(),
        vec![format!(
          "KnotData::notent: {shown} is not a known entity, class or tag \
           for KnotData. Use KnotData[] for a list of entities."
        )]
      );
    }
  }

  // Unknown properties stay unevaluated.
  #[test]
  fn knot_data_unknown_property() {
    assert_eq!(
      interpret(r#"KnotData["Trefoil", "NoSuchProperty"]"#).unwrap(),
      "KnotData[Trefoil, NoSuchProperty]"
    );
  }

  // KnotData["Properties"] includes the supported property names.
  #[test]
  fn knot_data_properties() {
    assert_eq!(
      interpret(
        r#"SubsetQ[KnotData["Properties"], {"AlexanderBriggsList",
             "AlexanderBriggsNotation", "CrossingNumber", "ImageData", "Name",
             "SpaceCurve", "StandardName"}]"#
      )
      .unwrap(),
      "True"
    );
  }

  // KnotData[name] renders the knot in 3D.
  #[test]
  fn knot_data_renders_graphics3d() {
    assert_eq!(
      interpret(r#"Head[KnotData["Trefoil"]]"#).unwrap(),
      "Graphics3D"
    );
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
