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
  fn element_data_all_returns_all_elements() {
    clear_state();
    let result = interpret("Length[ElementData[All]]").unwrap();
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
  fn element_data_meitnerium_melting_point_missing() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData["Meitnerium", "MeltingPoint"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
  }

  #[test]
  fn element_data_electron_config_tantalum() {
    clear_state();
    assert_eq!(
      interpret(r#"ElementData[73, "ElectronConfiguration"]"#).unwrap(),
      "{{2}, {2, 6}, {2, 6, 10}, {2, 6, 10, 14}, {2, 6, 3}, {2}}"
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

  #[test]
  fn element_data_absolute_boiling_point_helium() {
    // -268.93 °C + 273.15 = 4.22 K
    assert_eq!(
      interpret(r#"ElementData["He", "AbsoluteBoilingPoint"]"#).unwrap(),
      "4.22"
    );
  }

  #[test]
  fn element_data_absolute_melting_point_carbon() {
    assert_eq!(
      interpret(r#"ElementData["Carbon", "AbsoluteMeltingPoint"]"#).unwrap(),
      "3823.15"
    );
  }

  #[test]
  fn element_data_electronegativity_camelcase_alias() {
    // Both "Electronegativity" (lowercase n) and "ElectroNegativity"
    // (capital N) map to the same property.
    assert_eq!(
      interpret(r#"ElementData["He", "ElectroNegativity"]"#).unwrap(),
      "Missing[NotApplicable]"
    );
    assert_eq!(
      interpret(r#"ElementData["He", "Electronegativity"]"#).unwrap(),
      "Missing[NotApplicable]"
    );
  }

  #[test]
  fn element_data_electron_config_string_sulfur() {
    assert_eq!(
      interpret(r#"ElementData[16, "ElectronConfigurationString"]"#).unwrap(),
      "[Ne] 3s2 3p4"
    );
  }

  #[test]
  fn element_data_electron_config_string_iron() {
    assert_eq!(
      interpret(r#"ElementData["Iron", "ElectronConfigurationString"]"#)
        .unwrap(),
      "[Ar] 3d6 4s2"
    );
  }

  #[test]
  fn element_data_electron_config_string_hydrogen() {
    assert_eq!(
      interpret(r#"ElementData[1, "ElectronConfigurationString"]"#).unwrap(),
      "1s1"
    );
  }

  #[test]
  fn element_data_electron_config_string_helium() {
    assert_eq!(
      interpret(r#"ElementData["He", "ElectronConfigurationString"]"#).unwrap(),
      "1s2"
    );
  }

  #[test]
  fn element_data_recognised_property_without_data_returns_not_available() {
    // SpecificHeat, Density, IonizationEnergies, etc. are recognised by name
    // but not tabulated — return Missing[NotAvailable] rather than NotFound.
    assert_eq!(
      interpret(r#"ElementData["Tc", "SpecificHeat"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
    assert_eq!(
      interpret(r#"ElementData["Carbon", "IonizationEnergies"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
  }

  #[test]
  fn element_data_properties_full_list() {
    assert_eq!(
      interpret(r#"ElementData["Properties"]"#).unwrap(),
      "{Abbreviation, AbsoluteBoilingPoint, AbsoluteMeltingPoint, AtomicNumber, AtomicRadius, AtomicWeight, Block, BoilingPoint, BrinellHardness, BulkModulus, CovalentRadius, CrustAbundance, Density, DiscoveryYear, ElectroNegativity, ElectronAffinity, ElectronConfiguration, ElectronConfigurationString, ElectronShellConfiguration, FusionHeat, Group, IonizationEnergies, LiquidDensity, MeltingPoint, MohsHardness, Name, Period, PoissonRatio, Series, ShearModulus, SpecificHeat, StandardName, ThermalConductivity, VanDerWaalsRadius, VaporizationHeat, VickersHardness, YoungModulus}"
    );
  }
}
