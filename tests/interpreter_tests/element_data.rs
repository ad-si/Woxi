use super::*;

mod element_data_tests {
  use super::*;

  #[test]
  fn element_data_by_name_returns_entity() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon"]"#).unwrap(),
      "Entity[Element, Carbon]"
    );
  }

  #[test]
  fn element_data_by_number_returns_entity() {
    clear_state();
    assert_eq!(
      interpret("ElementData[6]").unwrap(),
      "Entity[Element, Carbon]"
    );
  }

  #[test]
  fn element_data_tungsten_by_number() {
    clear_state();
    assert_eq!(
      interpret("ElementData[74]").unwrap(),
      "Entity[Element, Tungsten]"
    );
  }

  #[test]
  fn element_data_no_args_returns_all_elements() {
    clear_state();
    let result = interpret("Length[ElementData[]]").unwrap();
    assert_eq!(result, "118");
  }

  #[test]
  fn element_data_atomic_number() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "AtomicNumber"]"#).unwrap(),
      "6"
    );
  }

  #[test]
  fn element_data_name() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "Name"]"#).unwrap(),
      "carbon"
    );
  }

  #[test]
  fn element_data_standard_name() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "StandardName"]"#).unwrap(),
      "Carbon"
    );
  }

  #[test]
  fn element_data_abbreviation() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "Abbreviation"]"#).unwrap(),
      "C"
    );
  }

  #[test]
  fn element_data_atomic_weight() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "AtomicWeight"]"#).unwrap(),
      "Quantity[12.011`5., AtomicMassUnit]"
    );
  }

  #[test]
  fn element_data_atomic_weight_hydrogen() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Hydrogen", "AtomicWeight"]"#).unwrap(),
      "Quantity[1.008`5., AtomicMassUnit]"
    );
  }

  #[test]
  fn element_data_group() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "Group"]"#).unwrap(),
      "14"
    );
  }

  #[test]
  fn element_data_period() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "Period"]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn element_data_block() {
    clear_state();
    assert_eq!(interpret(r#"ElementData["Carbon", "Block"]"#).unwrap(), "p");
  }

  #[test]
  fn element_data_electronegativity() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Hydrogen", "Electronegativity"]"#).unwrap(),
      "2.2`3."
    );
  }

  #[test]
  fn element_data_melting_point() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "MeltingPoint"]"#).unwrap(),
      "Quantity[3550.`4., DegreesCelsius]"
    );
  }

  #[test]
  fn element_data_boiling_point() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "BoilingPoint"]"#).unwrap(),
      "Quantity[4027.`4., DegreesCelsius]"
    );
  }

  #[test]
  fn element_data_atomic_radius() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "AtomicRadius"]"#).unwrap(),
      "Quantity[67.`2., Picometers]"
    );
  }

  #[test]
  fn element_data_electron_configuration() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Carbon", "ElectronConfiguration"]"#).unwrap(),
      "{{2}, {2, 2}}"
    );
  }

  #[test]
  fn element_data_lookup_by_abbreviation() {
    clear_state();
    assert_eq!(interpret(r#"ElementData["C", "Name"]"#).unwrap(), "carbon");
  }

  #[test]
  fn element_data_lookup_by_number() {
    clear_state();
    assert_eq!(interpret(r#"ElementData[6, "Name"]"#).unwrap(), "carbon");
  }

  #[test]
  fn element_data_iron_atomic_number() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Iron", "AtomicNumber"]"#).unwrap(),
      "26"
    );
  }

  #[test]
  fn element_data_iron_abbreviation() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Iron", "Abbreviation"]"#).unwrap(),
      "Fe"
    );
  }

  #[test]
  fn element_data_iron_melting_point() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Iron", "MeltingPoint"]"#).unwrap(),
      "Quantity[1538.`4., DegreesCelsius]"
    );
  }

  #[test]
  fn element_data_noble_gas_electronegativity() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Helium", "Electronegativity"]"#).unwrap(),
      r#"Missing[NotApplicable]"#
    );
  }

  #[test]
  fn element_data_lanthanide_group() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Lanthanum", "Group"]"#).unwrap(),
      r#"Missing[Undefined]"#
    );
  }

  #[test]
  fn element_data_properties_list() {
    clear_state();
    let result = interpret(r#"ElementData["Properties"]"#).unwrap();
    assert!(result.contains("AtomicNumber"));
    assert!(result.contains("AtomicWeight"));
    assert!(result.contains("MeltingPoint"));
  }

  #[test]
  fn element_data_gold_atomic_weight() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Gold", "AtomicWeight"]"#).unwrap(),
      "Quantity[196.96657`9., AtomicMassUnit]"
    );
  }

  #[test]
  fn element_data_technetium_atomic_weight() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Technetium", "AtomicWeight"]"#).unwrap(),
      "Quantity[97.`2., AtomicMassUnit]"
    );
  }

  #[test]
  fn element_data_lookup_by_fe() {
    clear_state();
    assert_eq!(interpret(r#"ElementData["Fe", "Name"]"#).unwrap(), "iron");
  }

  #[test]
  fn element_data_oganesson() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData[118, "StandardName"]"#).unwrap(),
      "Oganesson"
    );
  }

  #[test]
  fn element_data_missing_boiling_point() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Astatine", "BoilingPoint"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
  }

  #[test]
  fn element_data_electron_config_iron() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Iron", "ElectronConfiguration"]"#).unwrap(),
      "{{2}, {2, 6}, {2, 6, 6}, {2}}"
    );
  }
}
