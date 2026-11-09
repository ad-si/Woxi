use super::*;

mod chemical_data_tests {
  use super::*;

  // PubChem's atom order, bond list and depiction coordinates, as
  // wolframscript reports them.
  #[test]
  fn caffeine_structure_matches_pubchem_order() {
    clear_state();
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "VertexTypes"]"#).unwrap(),
      "{O, O, N, N, N, N, C, C, C, C, C, C, C, C, H, H, H, H, H, H, H, H, H, H}"
    );
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "EdgeRules"][[;; 5]]"#).unwrap(),
      "{1 -> 9, 2 -> 10, 3 -> 8, 3 -> 10, 3 -> 12}"
    );
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "EdgeTypes"][[;; 3]]"#).unwrap(),
      "{Double, Double, Single}"
    );
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "VertexCoordinates"][[;; 2]]"#)
        .unwrap(),
      "{{373.21, 200.}, {200., -100.}}"
    );
  }

  #[test]
  fn names_and_tallies() {
    clear_state();
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "Name"]"#).unwrap(),
      "caffeine"
    );
    assert_eq!(
      interpret(r#"ChemicalData["Caffeine", "ElementTally"]"#).unwrap(),
      "{{C, 8}, {H, 10}, {N, 4}, {O, 2}}"
    );
    assert_eq!(
      interpret(r#"ChemicalData[Entity["Chemical", "Butane"], "SMILES"]"#)
        .unwrap(),
      "CCCC"
    );
    assert_eq!(
      interpret(
        r#"ChemicalData["Butane", "CompoundFormulaDisplay"] ===
           Style[Row[{Subscript["C", 4], Subscript["H", 10]}],
             ScriptBaselineShifts -> {0.5, 0.75}]"#
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"Head[ChemicalData["Butane", "MolecularMass"]]"#).unwrap(),
      "Quantity"
    );
  }

  #[test]
  fn drawings_are_symbolic_graphics() {
    clear_state();
    assert_eq!(
      interpret(
        r#"Head /@ {ChemicalData["Water", "MoleculePlot"],
           ChemicalData["Water", "SpaceFillingMoleculePlot"],
           ChemicalData["Water", "StructureDiagram"], ChemicalData["Water"]}"#
      )
      .unwrap(),
      "{Graphics3D, Graphics3D, Graphics, Graphics}"
    );
    // One sphere per atom in the ball-and-stick plot.
    assert_eq!(
      interpret(
        r#"Length[Cases[ChemicalData["Water", "MoleculePlot"], _Sphere,
           Infinity]]"#
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn unknown_compound_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(r#"ChemicalData["NoSuchCompound", "Name"]"#).unwrap(),
      "ChemicalData[NoSuchCompound, Name]"
    );
  }

  #[test]
  fn unknown_property_is_missing() {
    clear_state();
    assert_eq!(
      interpret(r#"ChemicalData["Water", "NoSuchProperty"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
  }
}
