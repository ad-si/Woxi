use super::*;

mod finite_group_data_tests {
  use super::*;

  fn point_group(n: usize, property: &str) -> String {
    interpret(&format!(
      r#"FiniteGroupData[{{"CrystallographicPointGroup", {n}}}, "{property}"]"#
    ))
    .unwrap()
  }

  #[test]
  fn names_follow_the_schoenflies_symbol() {
    clear_state();
    assert_eq!(point_group(1, "Name"), "crystallographic point group C1");
    assert_eq!(point_group(27, "Name"), "crystallographic point group D6h");
    assert_eq!(point_group(30, "Name"), "crystallographic point group O");
    assert_eq!(point_group(31, "Name"), "crystallographic point group Td");
  }

  #[test]
  fn standard_name_is_the_specification() {
    clear_state();
    assert_eq!(
      interpret(r#"FiniteGroupData[{"CrystallographicPointGroup", 11}]"#)
        .unwrap(),
      "{CrystallographicPointGroup, 11}"
    );
    assert_eq!(
      point_group(11, "StandardName"),
      "{CrystallographicPointGroup, 11}"
    );
  }

  /// The family is no entity or class of its own, and it has 32 groups:
  /// either is reported and left as it is.
  #[test]
  fn family_name_and_index_out_of_range_are_not_entities() {
    clear_state();
    assert_eq!(
      interpret(r#"FiniteGroupData["CrystallographicPointGroup"]"#).unwrap(),
      "FiniteGroupData[CrystallographicPointGroup]"
    );
    assert_eq!(
      interpret(
        r#"FiniteGroupData[{"CrystallographicPointGroup", 33}, "Order"]"#
      )
      .unwrap(),
      "FiniteGroupData[{CrystallographicPointGroup, 33}, Order]"
    );
    clear_state();
    interpret(r#"FiniteGroupData[{"CrystallographicPointGroup", 0}]"#).unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "FiniteGroupData::notent: {CrystallographicPointGroup, 0} is not a \
         known entity, class or tag for FiniteGroupData. Use \
         FiniteGroupData[] for a list of entities."
      )),
      "{msgs:?}"
    );
  }

  #[test]
  fn properties_are_listed_and_unknown_ones_reported() {
    clear_state();
    assert_eq!(
      interpret(
        r#"{Length[#], Take[#, 3], MemberQ[#, "MultiplicationTable"]} &[FiniteGroupData["Properties"]]"#
      )
      .unwrap(),
      "{82, {Abelian, AlternateNames, AlternateStandardNames}, True}"
    );
    assert_eq!(
      interpret(
        r#"FiniteGroupData[{"CrystallographicPointGroup", 8}, "Properties"] === FiniteGroupData["Properties"]"#
      )
      .unwrap(),
      "True"
    );
    clear_state();
    assert_eq!(
      interpret(
        r#"FiniteGroupData[{"CrystallographicPointGroup", 8}, "Nope"]"#
      )
      .unwrap(),
      "FiniteGroupData[{CrystallographicPointGroup, 8}, Nope]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "FiniteGroupData::notprop: Nope is not a known property for \
         FiniteGroupData. Use FiniteGroupData[\"Properties\"] for a list of \
         properties."
      )),
      "{msgs:?}"
    );
  }

  #[test]
  fn orders_of_all_groups() {
    clear_state();
    assert_eq!(
      interpret(
        r#"Table[FiniteGroupData[{"CrystallographicPointGroup", i}, "Order"], {i, 32}]"#
      )
      .unwrap(),
      "{1, 2, 2, 2, 4, 4, 4, 8, 4, 4, 8, 8, 8, 8, 16, 3, 6, 6, 6, 12, 6, 6, 12, \
       12, 12, 12, 24, 12, 24, 24, 24, 48}"
    );
    assert_eq!(point_group(18, "Elements"), "{1, 2, 3, 4, 5, 6}");
  }

  #[test]
  fn crystal_classification() {
    clear_state();
    assert_eq!(
      interpret(
        r#"DeleteDuplicates[Table[FiniteGroupData[{"CrystallographicPointGroup", i}, "CrystalSystem"], {i, 32}]]"#
      )
      .unwrap(),
      "{Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic}"
    );
    assert_eq!(point_group(27, "CrystalSystem"), "Hexagonal");
    assert_eq!(point_group(27, "CrystalForm"), "dihexagonal dipyramidal");
    assert_eq!(point_group(27, "PointGroupType"), "{Centrosymmetric}");
    assert_eq!(point_group(9, "PointGroupType"), "{Enantiomorphic, Polar}");
    assert_eq!(point_group(10, "PointGroupType"), "{}");
  }

  #[test]
  fn several_properties_at_once() {
    clear_state();
    assert_eq!(
      interpret(
        r#"FiniteGroupData[{"CrystallographicPointGroup", 32}, {"Name", "CrystalSystem", "Order"}]"#
      )
      .unwrap(),
      "{crystallographic point group Oh, Cubic, 48}"
    );
  }

  #[test]
  fn small_multiplication_tables() {
    clear_state();
    assert_eq!(point_group(1, "MultiplicationTable"), "{{1}}");
    assert_eq!(point_group(2, "MultiplicationTable"), "{{1, 2}, {2, 1}}");
    assert_eq!(
      point_group(9, "MultiplicationTable"),
      "{{1, 2, 3, 4}, {2, 3, 4, 1}, {3, 4, 1, 2}, {4, 1, 2, 3}}"
    );
    assert_eq!(
      point_group(6, "MultiplicationTable"),
      "{{1, 2, 3, 4}, {2, 1, 4, 3}, {3, 4, 1, 2}, {4, 3, 2, 1}}"
    );
    // Rotations first, then the reflected coset.
    assert_eq!(
      point_group(18, "MultiplicationTable"),
      "{{1, 2, 3, 4, 5, 6}, {2, 3, 1, 6, 4, 5}, {3, 1, 2, 5, 6, 4}, \
       {4, 5, 6, 1, 2, 3}, {5, 6, 4, 3, 1, 2}, {6, 4, 5, 2, 3, 1}}"
    );
  }

  // The layout of the D6h table: the hexagonal rotations, their coset
  // under a twofold axis, then both again multiplied by the inversion.
  #[test]
  fn d6h_multiplication_table_blocks() {
    clear_state();
    assert_eq!(
      interpret(
        r#"t = FiniteGroupData[{"CrystallographicPointGroup", 27}, "MultiplicationTable"]; {t[[2]], t[[7]], t[[8, 7]], t[[13, 13]], t[[24]]}"#
      )
      .unwrap(),
      "{{2, 3, 4, 5, 6, 1, 12, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 13, 24, \
       19, 20, 21, 22, 23}, {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 19, 20, \
       21, 22, 23, 24, 13, 14, 15, 16, 17, 18}, 6, 1, {24, 19, 20, 21, 22, \
       23, 14, 15, 16, 17, 18, 13, 12, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 1}}"
    );
  }

  // Every table is a group table with the identity first.
  #[test]
  fn every_table_satisfies_the_group_axioms() {
    clear_state();
    assert_eq!(
      interpret(
        r#"ok[t_] := With[{n = Length[t]}, And[First[t] === Range[n], t[[All, 1]] === Range[n], AllTrue[t, Sort[#] === Range[n] &], AllTrue[Transpose[t], Sort[#] === Range[n] &], And @@ Flatten[Table[t[[t[[a, b]], c]] == t[[a, t[[b, c]]]], {a, n}, {b, n}, {c, n}]]]]; And @@ Table[ok[FiniteGroupData[{"CrystallographicPointGroup", i}, "MultiplicationTable"]], {i, 32}]"#
      )
      .unwrap(),
      "True"
    );
  }

  // Isomorphism invariants: which groups are Abelian, and how many
  // elements square to the identity.
  #[test]
  fn tables_have_the_right_structure() {
    clear_state();
    assert_eq!(
      interpret(
        r#"Select[Range[32], With[{t = FiniteGroupData[{"CrystallographicPointGroup", #}, "MultiplicationTable"]}, t =!= Transpose[t]] &]"#
      )
      .unwrap(),
      "{12, 13, 14, 15, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32}"
    );
    assert_eq!(
      interpret(
        r#"Table[Count[Diagonal[FiniteGroupData[{"CrystallographicPointGroup", i}, "MultiplicationTable"]], 1], {i, {10, 14, 22, 28, 29, 30, 32}}]"#
      )
      .unwrap(),
      "{2, 6, 2, 4, 8, 10, 20}"
    );
  }

  #[test]
  fn cayley_graphs_are_connected() {
    clear_state();
    assert_eq!(
      interpret(
        r#"g = FiniteGroupData[{"CrystallographicPointGroup", 30}, "CayleyGraph"]; {Head[g], VertexCount[g], EdgeCount[g], ConnectedGraphQ[g]}"#
      )
      .unwrap(),
      "{Graph, 24, 72, True}"
    );
    assert_eq!(
      interpret(
        r#"And @@ Table[ConnectedGraphQ[FiniteGroupData[{"CrystallographicPointGroup", i}, "CayleyGraph"]], {i, 32}]"#
      )
      .unwrap(),
      "True"
    );
    // Each generator's edges are drawn in a color of their own.
    let svg = interpret(
      r#"ExportString[Show[FiniteGroupData[{"CrystallographicPointGroup", 27}, "CayleyGraph"], ImageSize -> {450, 450}], "SVG"]"#,
    )
    .unwrap();
    for color in ["rgb(94,129,181)", "rgb(225,156,36)", "rgb(143,176,50)"] {
      assert!(svg.contains(color), "missing edge color {color}");
    }
  }

  #[test]
  fn table_can_be_plotted() {
    clear_state();
    let svg = interpret(
      r#"ExportString[ArrayPlot[FiniteGroupData[{"CrystallographicPointGroup", 27}, "MultiplicationTable"], ColorFunction -> "RedBlueTones", Frame -> False], "SVG"]"#,
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
  }

  #[test]
  fn unknown_groups_and_properties_stay_unevaluated() {
    clear_state();
    assert_eq!(
      point_group(33, "Order"),
      "FiniteGroupData[{CrystallographicPointGroup, 33}, Order]"
    );
    assert_eq!(
      point_group(3, "NoSuchProperty"),
      "FiniteGroupData[{CrystallographicPointGroup, 3}, NoSuchProperty]"
    );
    assert_eq!(point_group(5, "ElementNames"), "Missing[NotAvailable]");
  }
}
