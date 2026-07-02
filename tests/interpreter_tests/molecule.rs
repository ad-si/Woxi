use super::*;

mod molecule_tests {
  use super::*;

  // --- Construction from chemical names ---------------------------------

  #[test]
  fn molecule_from_name_water() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["water"]"#).unwrap(),
      "Molecule[{Atom[O], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single]}]"
    );
  }

  #[test]
  fn molecule_name_lookup_is_case_insensitive() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["Water"]"#).unwrap(),
      interpret(r#"Molecule["water"]"#).unwrap()
    );
  }

  #[test]
  fn molecule_from_name_methane() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["methane"]"#).unwrap(),
      "Molecule[{Atom[C], Atom[H], Atom[H], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single], \
       Bond[{1, 5}, Single]}]"
    );
  }

  // --- Construction from SMILES strings ---------------------------------

  #[test]
  fn molecule_from_smiles_ethanol() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["CCO"]"#).unwrap(),
      "Molecule[{Atom[C], Atom[C], Atom[O], Atom[H], Atom[H], Atom[H], \
       Atom[H], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{2, 3}, Single], Bond[{1, 4}, Single], \
       Bond[{1, 5}, Single], Bond[{1, 6}, Single], Bond[{2, 7}, Single], \
       Bond[{2, 8}, Single], Bond[{3, 9}, Single]}]"
    );
  }

  #[test]
  fn molecule_from_smiles_double_bond() {
    clear_state();
    // formaldehyde
    assert_eq!(
      interpret(r#"Molecule["C=O"]"#).unwrap(),
      "Molecule[{Atom[C], Atom[O], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Double], Bond[{1, 3}, Single], Bond[{1, 4}, Single]}]"
    );
  }

  #[test]
  fn molecule_from_smiles_triple_bond() {
    clear_state();
    // hydrogen cyanide
    assert_eq!(
      interpret(r#"Molecule["C#N"]"#).unwrap(),
      "Molecule[{Atom[C], Atom[N], Atom[H]}, \
       {Bond[{1, 2}, Triple], Bond[{1, 3}, Single]}]"
    );
  }

  #[test]
  fn molecule_from_smiles_branches() {
    clear_state();
    // isobutane: central carbon with three methyl groups
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["CC(C)C"], "MolecularFormula"]"#)
        .unwrap(),
      "C4H10"
    );
  }

  #[test]
  fn molecule_from_smiles_ring() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["C1CCCCC1"], "MolecularFormula"]"#)
        .unwrap(),
      "C6H12"
    );
  }

  #[test]
  fn molecule_from_smiles_percent_ring_closure() {
    clear_state();
    // cyclododecane written with a two-digit ring bond number
    assert_eq!(
      interpret(
        r#"MoleculeValue[Molecule["C%12CCCCCCCCCCC%12"], "MolecularFormula"]"#
      )
      .unwrap(),
      "C12H24"
    );
  }

  #[test]
  fn molecule_from_aromatic_smiles() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["c1ccccc1"]"#).unwrap(),
      "Molecule[{Atom[C], Atom[C], Atom[C], Atom[C], Atom[C], Atom[C], \
       Atom[H], Atom[H], Atom[H], Atom[H], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Aromatic], Bond[{2, 3}, Aromatic], \
       Bond[{3, 4}, Aromatic], Bond[{4, 5}, Aromatic], \
       Bond[{5, 6}, Aromatic], Bond[{1, 6}, Aromatic], \
       Bond[{1, 7}, Single], Bond[{2, 8}, Single], Bond[{3, 9}, Single], \
       Bond[{4, 10}, Single], Bond[{5, 11}, Single], \
       Bond[{6, 12}, Single]}]"
    );
  }

  #[test]
  fn aromatic_nitrogen_gets_no_hydrogen() {
    clear_state();
    // pyridine: C5H5N — the ring nitrogen carries no hydrogen
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["c1ccncc1"], "MolecularFormula"]"#)
        .unwrap(),
      "C5H5N"
    );
  }

  #[test]
  fn bracket_hydrogen_count_is_respected() {
    clear_state();
    // pyrrole: the [nH] nitrogen carries exactly one hydrogen
    assert_eq!(
      interpret(
        r#"MoleculeValue[Molecule["c1cc[nH]c1"], "MolecularFormula"]"#
      )
      .unwrap(),
      "C4H5N"
    );
  }

  #[test]
  fn molecule_from_smiles_formal_charge() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["[NH4+]"]"#).unwrap(),
      "Molecule[{Atom[N, FormalCharge -> 1], Atom[H], Atom[H], Atom[H], \
       Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single], \
       Bond[{1, 5}, Single]}]"
    );
  }

  #[test]
  fn molecule_from_smiles_mass_number() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["[13C]"]"#).unwrap(),
      "Molecule[{Atom[C, MassNumber -> 13]}, {}]"
    );
  }

  #[test]
  fn molecule_from_smiles_disconnected_components() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["[Na+].[Cl-]"]"#).unwrap(),
      "Molecule[{Atom[Na, FormalCharge -> 1], \
       Atom[Cl, FormalCharge -> -1]}, {}]"
    );
  }

  #[test]
  fn multi_character_charge_forms() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["[Fe+2]"]"#).unwrap(),
      "Molecule[{Atom[Fe, FormalCharge -> 2]}, {}]"
    );
    assert_eq!(
      interpret(r#"Molecule["[Fe++]"]"#).unwrap(),
      "Molecule[{Atom[Fe, FormalCharge -> 2]}, {}]"
    );
  }

  // --- Construction from explicit atom and bond lists -------------------

  #[test]
  fn molecule_from_atom_and_bond_lists_fills_valences() {
    clear_state();
    // methanol: hydrogens are added to fill the normal valences
    assert_eq!(
      interpret(r#"Molecule[{"C", "O"}, {Bond[{1, 2}, "Single"]}]"#).unwrap(),
      "Molecule[{Atom[C], Atom[O], Atom[H], Atom[H], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single], \
       Bond[{1, 5}, Single], Bond[{2, 6}, Single]}]"
    );
  }

  #[test]
  fn molecule_accepts_atomic_numbers() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule[{6, 8}, {Bond[{1, 2}, "Single"]}]"#).unwrap(),
      interpret(r#"Molecule[{"C", "O"}, {Bond[{1, 2}, "Single"]}]"#).unwrap()
    );
  }

  #[test]
  fn bond_without_type_defaults_to_single() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule[{"C", "O"}, {Bond[{1, 2}]}]"#).unwrap(),
      interpret(r#"Molecule[{"C", "O"}, {Bond[{1, 2}, "Single"]}]"#).unwrap()
    );
  }

  #[test]
  fn molecule_evaluation_is_idempotent() {
    clear_state();
    let canonical = interpret(r#"Molecule["water"]"#).unwrap();
    assert_eq!(interpret(&canonical).unwrap(), canonical);
  }

  // --- Invalid input stays unevaluated -----------------------------------

  #[test]
  fn uninterpretable_string_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["gibberish"]"#).unwrap(),
      "Molecule[gibberish]"
    );
  }

  #[test]
  fn unclosed_ring_bond_stays_unevaluated() {
    clear_state();
    assert_eq!(interpret(r#"Molecule["C1CC"]"#).unwrap(), "Molecule[C1CC]");
  }

  #[test]
  fn out_of_range_bond_index_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule[{"C"}, {Bond[{1, 5}, "Single"]}]"#).unwrap(),
      "Molecule[{C}, {Bond[{1, 5}, Single]}]"
    );
  }

  #[test]
  fn unknown_element_symbol_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule[{"Xx"}, {}]"#).unwrap(),
      "Molecule[{Xx}, {}]"
    );
  }

  // --- MoleculeQ ---------------------------------------------------------

  #[test]
  fn molecule_q_true_for_valid_molecule() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeQ[Molecule["ethanol"]]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn molecule_q_false_for_non_molecule() {
    clear_state();
    assert_eq!(interpret("MoleculeQ[5]").unwrap(), "False");
    assert_eq!(interpret(r#"MoleculeQ["water"]"#).unwrap(), "False");
  }

  #[test]
  fn molecule_q_false_for_invalid_molecule() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeQ[Molecule["gibberish"]]"#).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(r#"MoleculeQ[Molecule[{Atom["O"]}, {Bond[{1, 2}]}]]"#)
        .unwrap(),
      "False"
    );
  }

  // --- AtomList and BondList ---------------------------------------------

  #[test]
  fn atom_list_returns_atoms() {
    clear_state();
    assert_eq!(
      interpret(r#"AtomList[Molecule["methane"]]"#).unwrap(),
      "{Atom[C], Atom[H], Atom[H], Atom[H], Atom[H]}"
    );
  }

  #[test]
  fn bond_list_returns_bonds() {
    clear_state();
    assert_eq!(
      interpret(r#"BondList[Molecule["ammonia"]]"#).unwrap(),
      "{Bond[{1, 2}, Single], Bond[{1, 3}, Single], Bond[{1, 4}, Single]}"
    );
  }

  #[test]
  fn atom_list_of_non_molecule_stays_unevaluated() {
    clear_state();
    assert_eq!(interpret("AtomList[5]").unwrap(), "AtomList[5]");
  }

  // --- MoleculeValue and property access ---------------------------------

  #[test]
  fn molecule_value_atom_count() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["ethanol"], "AtomCount"]"#)
        .unwrap(),
      "9"
    );
  }

  #[test]
  fn molecule_value_bond_count() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["water"], "BondCount"]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn molecule_value_molecular_formula() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["caffeine"], "MolecularFormula"]"#)
        .unwrap(),
      "C8H10N4O2"
    );
  }

  #[test]
  fn molecule_value_net_charge() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["[NH4+]"], "NetCharge"]"#).unwrap(),
      "1"
    );
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["water"], "NetCharge"]"#).unwrap(),
      "0"
    );
  }

  #[test]
  fn molecule_value_list_of_properties() {
    clear_state();
    assert_eq!(
      interpret(
        r#"MoleculeValue[Molecule["glucose"], {"AtomCount", "MolecularFormula"}]"#
      )
      .unwrap(),
      "{24, C6H12O6}"
    );
  }

  #[test]
  fn molecule_subvalue_property_access() {
    clear_state();
    assert_eq!(
      interpret(r#"Molecule["water"]["AtomCount"]"#).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(r#"Molecule["benzene"]["MolecularFormula"]"#).unwrap(),
      "C6H6"
    );
  }

  #[test]
  fn unknown_property_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["water"], "Bogus"]"#).unwrap(),
      "MoleculeValue[Molecule[{Atom[O], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single]}], Bogus]"
    );
  }

  // --- Formula edge cases --------------------------------------------------

  #[test]
  fn formula_without_carbon_is_alphabetical() {
    clear_state();
    // Hill order without carbon: strictly alphabetical (H before N and O)
    assert_eq!(
      interpret(
        r#"MoleculeValue[Molecule["nitric acid"], "MolecularFormula"]"#
      )
      .unwrap(),
      "HNO3"
    );
  }

  #[test]
  fn formula_of_charged_molecule_shows_net_charge() {
    clear_state();
    assert_eq!(
      interpret(r#"MoleculeValue[Molecule["[NH4+]"], "MolecularFormula"]"#)
        .unwrap(),
      "H4N+"
    );
  }
}
