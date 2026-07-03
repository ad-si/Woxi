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
      interpret(r#"MoleculeValue[Molecule["c1cc[nH]c1"], "MolecularFormula"]"#)
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
      interpret(r#"MoleculeQ[Molecule[{Atom["O"]}, {Bond[{1, 2}]}]]"#).unwrap(),
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
      interpret(r#"MoleculeValue[Molecule["ethanol"], "AtomCount"]"#).unwrap(),
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
    assert_eq!(interpret(r#"Molecule["water"]["AtomCount"]"#).unwrap(), "3");
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

  // --- Molecule information tile -------------------------------------------

  #[test]
  fn molecule_exports_info_tile() {
    clear_state();
    // ExportString[Molecule[…], "SVG"] draws the information tile: a structure
    // thumbnail beside the formula, atom count, and bond count — not a text
    // dump of the symbolic `Molecule[…]` expression.
    let svg = interpret(r#"ExportString[Molecule["water"], "SVG"]"#).unwrap();
    assert!(svg.starts_with("<svg"), "should be an SVG document");
    assert!(!svg.contains("Molecule"), "must not dump the symbolic form");
    assert!(svg.contains("Formula: "), "shows the molecular formula");
    assert!(svg.contains("Atoms: "), "shows the atom count");
    assert!(svg.contains("Bonds: "), "shows the bond count");
    // Water counts: 3 atoms (O + 2 H) and 2 bonds, shown as value tspans.
    assert!(svg.contains(">3</tspan>"));
    assert!(svg.contains(">2</tspan>"));
    // The thumbnail draws the two O–H bonds and labels the oxygen.
    assert_eq!(svg.matches("<line").count(), 2);
    assert!(svg.contains(">O</text>"), "thumbnail labels the oxygen");
  }

  #[test]
  fn tile_formula_uses_subscripts_and_charge() {
    clear_state();
    // Ammonium: formula H4N+ — the element count "4" is a subscript tspan and
    // the "+" net charge rides as a superscript tspan.
    let svg = interpret(r#"ExportString[Molecule["[NH4+]"], "SVG"]"#).unwrap();
    assert!(svg.contains(">H</tspan>"));
    assert!(svg.contains(">4</tspan>"), "element count as a tspan");
    assert!(svg.contains(">+</tspan>"), "net charge as a tspan");
  }

  #[test]
  fn molecule_tile_in_visual_mode() {
    use woxi::interpret_with_stdout;
    clear_state();
    // In a visual host (playground / woxi-studio) a bare Molecule result is
    // captured as the info tile rather than echoed symbolically.
    let r = interpret_with_stdout(r#"Molecule["ethanol"]"#).unwrap();
    let svg = r.graphics.expect("Molecule should produce a graphics SVG");
    assert!(svg.contains("Formula: "));
    assert!(svg.contains("Atoms: "));
  }

  #[test]
  fn bare_molecule_stays_symbolic_in_cli() {
    clear_state();
    // Plain `interpret` (CLI / wolframscript parity) keeps the symbolic echo.
    assert_eq!(
      interpret(r#"Molecule["water"]"#).unwrap(),
      "Molecule[{Atom[O], Atom[H], Atom[H]}, \
       {Bond[{1, 2}, Single], Bond[{1, 3}, Single]}]"
    );
  }

  // --- MoleculePlot structure diagram --------------------------------------

  #[test]
  fn molecule_plot_is_graphics_in_cli() {
    clear_state();
    // Like other plot functions, MoleculePlot echoes as -Graphics- in the CLI.
    assert_eq!(
      interpret(r#"MoleculePlot[Molecule["water"]]"#).unwrap(),
      "-Graphics-"
    );
    // It also accepts a bare name / SMILES specification.
    assert_eq!(
      interpret(r#"MoleculePlot["benzene"]"#).unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn molecule_plot_draws_aromatic_ring() {
    clear_state();
    let svg =
      interpret(r#"ExportString[MoleculePlot["benzene"], "SVG"]"#).unwrap();
    // Six ring bonds, each aromatic bond adding an inner line: 12 strokes.
    assert_eq!(svg.matches("<line").count(), 12);
    // A pure-carbon aromatic ring has no atom labels.
    assert!(!svg.contains("</text>"));
  }

  #[test]
  fn molecule_plot_draws_double_bonds() {
    clear_state();
    // O=C=O: two double bonds, each drawn as two parallel strokes.
    let svg =
      interpret(r#"ExportString[MoleculePlot["carbon dioxide"], "SVG"]"#)
        .unwrap();
    assert_eq!(svg.matches("<line").count(), 4);
    assert_eq!(svg.matches(">O</text>").count(), 2);
  }

  #[test]
  fn molecule_plot_labels_heteroatoms() {
    clear_state();
    // The hydroxyl oxygen of ethanol is labeled in the structure diagram.
    let svg =
      interpret(r#"ExportString[MoleculePlot["ethanol"], "SVG"]"#).unwrap();
    assert!(svg.contains(">O</text>"));
  }
}
