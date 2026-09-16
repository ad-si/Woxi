use super::*;

mod isotope_data_tests {
  use super::super::case_helpers::assert_case;
  use super::*;

  #[test]
  fn isotope_data_by_atomic_number_lists_isotopes() {
    clear_state();
    assert_eq!(
      interpret("IsotopeData[6]").unwrap(),
      "{Entity[Isotope, Carbon12], Entity[Isotope, Carbon13], \
       Entity[Isotope, Carbon14]}"
    );
  }

  #[test]
  fn isotope_data_by_element_name_lists_isotopes() {
    clear_state();
    assert_eq!(
      interpret(r#"IsotopeData["Hydrogen"]"#).unwrap(),
      "{Entity[Isotope, Hydrogen1], Entity[Isotope, Hydrogen2], \
       Entity[Isotope, Hydrogen3]}"
    );
  }

  #[test]
  fn isotope_data_element_massnumber_pair() {
    clear_state();
    assert_eq!(
      interpret(r#"IsotopeData[{"Carbon", 12}]"#).unwrap(),
      "Entity[Isotope, Carbon12]"
    );
  }

  #[test]
  fn isotope_data_mass_number_and_atomic_number() {
    clear_state();
    assert_eq!(
      interpret(
        r#"{IsotopeData[Entity["Isotope", "Carbon12"], "MassNumber"],
            IsotopeData[Entity["Isotope", "Carbon12"], "AtomicNumber"],
            IsotopeData[Entity["Isotope", "Carbon12"], "NeutronNumber"]}"#
      )
      .unwrap(),
      "{12, 6, 6}"
    );
  }

  #[test]
  fn isotope_data_binding_energy_carbon12() {
    clear_state();
    // Textbook value: the total nuclear binding energy of carbon-12 is
    // 92.16 MeV.
    assert_eq!(
      interpret(
        r#"Round[QuantityMagnitude[
             IsotopeData[Entity["Isotope", "Carbon12"], "BindingEnergy"]],
           0.01]"#
      )
      .unwrap(),
      "92.16"
    );
  }

  #[test]
  fn isotope_data_abundance_known_and_missing() {
    clear_state();
    assert_eq!(
      interpret(
        r#"{IsotopeData[Entity["Isotope", "Carbon12"], "IsotopeAbundance"],
            IsotopeData[Entity["Isotope", "Carbon14"], "IsotopeAbundance"]}"#
      )
      .unwrap(),
      "{0.9893, Missing[NotAvailable]}"
    );
  }

  #[test]
  fn isotope_data_atomic_mass_is_a_dalton_quantity() {
    clear_state();
    assert_eq!(
      interpret(r#"IsotopeData[Entity["Isotope", "Carbon12"], "AtomicMass"]"#)
        .unwrap(),
      "Quantity[12., Daltons]"
    );
  }

  #[test]
  fn isotope_data_standard_name() {
    clear_state();
    assert_eq!(
      interpret(
        r#"IsotopeData[Entity["Isotope", "Carbon12"], "StandardName"]"#
      )
      .unwrap(),
      "Carbon12"
    );
  }

  #[test]
  fn isotope_data_properties_list() {
    clear_state();
    assert_eq!(
      interpret("IsotopeData[\"Properties\"]").unwrap(),
      "{AtomicMass, AtomicNumber, BindingEnergy, IsotopeAbundance, \
       MassNumber, NeutronNumber, StandardName}"
    );
  }

  #[test]
  fn isotope_data_preload_is_a_noop() {
    assert_case(r#"IsotopeData[All, "Preload"]"#, "Null");
  }

  #[test]
  fn isotope_data_all_matches_no_arg_count() {
    clear_state();
    assert_eq!(
      interpret("Length[IsotopeData[All]] == Length[IsotopeData[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn element_data_stable_isotopes_present_and_absent() {
    clear_state();
    // Carbon has two naturally occurring isotopes; polonium (Z=84) has none.
    assert_eq!(
      interpret(
        r#"{ElementData[6, "StableIsotopes"], ElementData[84, "StableIsotopes"]}"#
      )
      .unwrap(),
      "{{Entity[Isotope, Carbon12], Entity[Isotope, Carbon13]}, {}}"
    );
  }

  #[test]
  fn list_line_plot_strips_quantity_magnitude() {
    clear_state();
    // A Quantity-valued y-coordinate must still produce a real plotted
    // curve rather than being silently dropped.
    assert_eq!(
      interpret(
        r#"ListLinePlot[{{1, Quantity[2.0, "Meters"]},
                          {2, Quantity[3.0, "Meters"]}}] === ListLinePlot[{{1, 2.0}, {2, 3.0}}]"#
      )
      .unwrap(),
      "True"
    );
  }
}
