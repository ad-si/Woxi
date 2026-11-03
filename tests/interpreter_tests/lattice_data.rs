use super::*;

mod lattice_data_tests {
  use super::*;

  // Only wolframscript's property names are properties, and only its
  // entity names (no abbreviations) are entities.
  #[test]
  fn unknown_properties_and_abbreviations() {
    clear_state();
    let r = interpret_with_stdout(
      r#"{LatticeData["SimpleCubic", "CoordinationNumber"], LatticeData["BCC", "Image"]}"#,
    )
    .unwrap();
    assert_eq!(
      r.result,
      "{LatticeData[SimpleCubic, CoordinationNumber], LatticeData[BCC, Image]}"
    );
    assert!(r.warnings.iter().any(|w| w.contains(
      "LatticeData::notprop: CoordinationNumber is not a known property for LatticeData."
    )));
    assert!(r.warnings.iter().any(|w| w.contains(
      "LatticeData::notent: BCC is not a known entity, class or tag for LatticeData."
    )));
    assert_eq!(
      interpret(r#"{Length[LatticeData[]], Length[LatticeData["Properties"]], MemberQ[LatticeData[], "FaceCenteredCubic"]}"#)
        .unwrap(),
      "{21, 38, True}"
    );
  }

  #[test]
  fn image_is_graphics3d() {
    assert_eq!(
      interpret(r#"Head[LatticeData["FaceCenteredCubic", "Image"]]"#).unwrap(),
      "Graphics3D"
    );
  }

  // The generating vectors (GeneralizedWatermanPolyhedra).
  #[test]
  fn basis() {
    assert_eq!(
      interpret(
        r#"{Head[LatticeData["SimpleCubic", "Basis"]], Normal[LatticeData["SimpleCubic", "Basis"]], LatticeData["BodyCenteredCubic", "Basis"], LatticeData["FaceCenteredCubic", "Basis"]}"#
      )
      .unwrap(),
      "{SparseArray, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{2, 0, 0}, {0, 2, 0}, {1, 1, 1}}, {{-1, -1, 0}, {1, -1, 0}, {0, 1, -1}}}"
    );
  }

  #[test]
  fn unknown_lattice_stays_unevaluated() {
    assert_eq!(
      interpret(r#"LatticeData["Nope", "Image"]"#).unwrap(),
      "LatticeData[Nope, Image]"
    );
  }
}
