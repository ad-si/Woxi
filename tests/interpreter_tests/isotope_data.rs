use super::*;

mod isotope_data_tests {
  use super::super::case_helpers::assert_case;
  use super::*;

  // Woxi's bundled table is NIST's, narrower than Wolfram's full nuclide
  // chart (see conformance_gaps.md), so the assertion is that every
  // bundled isotope of the element is in the answer, in mass-number order
  // — a statement both engines agree on, unlike the raw list.
  #[test]
  fn isotope_data_by_atomic_number_lists_isotopes() {
    clear_state();
    assert_eq!(
      interpret(
        "SubsetQ[IsotopeData[6], {Entity[\"Isotope\", \"Carbon12\"], \
Entity[\"Isotope\", \"Carbon13\"], Entity[\"Isotope\", \"Carbon14\"]}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        r#"IsotopeData[6, "MassNumber"] == Sort[IsotopeData[6, "MassNumber"]]"#
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn isotope_data_by_element_name_lists_isotopes() {
    clear_state();
    assert_eq!(
      interpret(
        "SubsetQ[IsotopeData[\"Hydrogen\"], \
{Entity[\"Isotope\", \"Hydrogen1\"], Entity[\"Isotope\", \"Hydrogen2\"], \
Entity[\"Isotope\", \"Hydrogen3\"]}]"
      )
      .unwrap(),
      "True"
    );
  }

  // The pair form is `{atomicNumber, massNumber}`; an element *name* paired
  // with a mass number is not a known entity and stays unevaluated, exactly
  // as in wolframscript.
  #[test]
  fn isotope_data_atomic_number_massnumber_pair() {
    clear_state();
    assert_eq!(
      interpret("IsotopeData[{6, 12}]").unwrap(),
      "Entity[Isotope, Carbon12]"
    );
  }

  #[test]
  fn isotope_data_element_name_pair_is_not_an_entity() {
    clear_state();
    assert_eq!(
      interpret(r#"Quiet[IsotopeData[{"Carbon", 12}]]"#).unwrap(),
      "IsotopeData[{Carbon, 12}]"
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

  // "BindingEnergy" is the binding energy *per nucleon*, as in
  // wolframscript: carbon-12's textbook 92.16 MeV total over 12 nucleons is
  // 7.68 MeV, and helium-4's 28.30 MeV total is 7.07 MeV.
  #[test]
  fn isotope_data_binding_energy_is_per_nucleon() {
    clear_state();
    assert_eq!(
      interpret(
        r#"Round[QuantityMagnitude[
             IsotopeData[#, "BindingEnergy"]], 0.01] & /@
           {Entity["Isotope", "Carbon12"], Entity["Isotope", "Helium4"]}"#
      )
      .unwrap(),
      "{7.68, 7.07}"
    );
  }

  // A natural abundance is a percentage, and an isotope that does not occur
  // naturally has an exact `0 Percent` abundance rather than a `Missing`.
  #[test]
  fn isotope_data_abundance_is_a_percentage() {
    clear_state();
    assert_eq!(
      interpret(
        r#"{IsotopeData[Entity["Isotope", "Carbon12"], "IsotopeAbundance"],
            IsotopeData[Entity["Isotope", "Carbon14"], "IsotopeAbundance"]}"#
      )
      .unwrap(),
      "{Quantity[98.93, Percent], Quantity[0, Percent]}"
    );
  }

  #[test]
  fn isotope_data_atomic_mass_is_an_atomic_mass_unit_quantity() {
    clear_state();
    assert_eq!(
      interpret(
        r#"{QuantityMagnitude[
              IsotopeData[Entity["Isotope", "Carbon12"], "AtomicMass"]],
            QuantityUnit[
              IsotopeData[Entity["Isotope", "Carbon12"], "AtomicMass"]]}"#
      )
      .unwrap(),
      "{12., AtomicMassUnit}"
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

  // Properties are named by `EntityProperty["Isotope", …]` objects. Wolfram
  // lists every property of its curated chart; Woxi lists the ones it
  // answers, so the shared assertion is that its list is a subset.
  #[test]
  fn isotope_data_properties_list() {
    clear_state();
    assert_eq!(
      interpret("IsotopeData[\"Properties\"]").unwrap(),
      "{EntityProperty[Isotope, AtomicMass], \
       EntityProperty[Isotope, AtomicNumber], \
       EntityProperty[Isotope, BindingEnergy], \
       EntityProperty[Isotope, IsotopeAbundance], \
       EntityProperty[Isotope, MassNumber], \
       EntityProperty[Isotope, NeutronNumber], \
       EntityProperty[Isotope, StandardName]}"
    );
  }

  #[test]
  fn isotope_data_properties_are_entity_properties_of_isotope() {
    clear_state();
    assert_eq!(
      interpret(
        "SubsetQ[IsotopeData[\"Properties\"], \
{EntityProperty[\"Isotope\", \"AtomicMass\"], \
EntityProperty[\"Isotope\", \"BindingEnergy\"], \
EntityProperty[\"Isotope\", \"MassNumber\"]}]"
      )
      .unwrap(),
      "True"
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

  // An element or `All` names a *class* of isotopes: the property is mapped
  // over its members, in the order the class lists them.
  #[test]
  fn isotope_data_property_maps_over_a_class() {
    clear_state();
    // Woxi's carbon isotopes are the NIST subset {12, 13, 14} of Wolfram's
    // full {8, …, 23} chart, so the shared assertion is the subset.
    assert_eq!(
      interpret(r#"SubsetQ[IsotopeData[6, "MassNumber"], {12, 13, 14}]"#)
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        r#"Length[IsotopeData[All, "MassNumber"]] == Length[IsotopeData[All]]"#
      )
      .unwrap(),
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
