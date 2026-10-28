use super::*;

mod lattice_data_tests {
  use super::*;

  #[test]
  fn coordination_numbers() {
    assert_eq!(
      interpret(
        r#"Map[LatticeData[#, "CoordinationNumber"] &, {"SC", "BCC", "FCC"}]"#
      )
      .unwrap(),
      "{6, 8, 12}"
    );
  }

  #[test]
  fn unit_cell_point_counts() {
    assert_eq!(
      interpret(
        r#"Map[Length[LatticeData[#, "LatticePoints"]] &, {"SimpleCubic", "BodyCenteredCubic", "FaceCenteredCubic"}]"#
      )
      .unwrap(),
      "{8, 9, 14}"
    );
  }

  #[test]
  fn image_is_graphics3d_usable_in_show() {
    assert_eq!(
      interpret(r#"Head[LatticeData["FaceCenteredCubic", "Image"]]"#).unwrap(),
      "Graphics3D"
    );
    assert_eq!(
      interpret(
        r#"Head[Show[LatticeData["BCC", "Image"], Graphics3D[{Red, Point[{0, 0, 0}]}]]]"#
      )
      .unwrap(),
      "Graphics3D"
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
