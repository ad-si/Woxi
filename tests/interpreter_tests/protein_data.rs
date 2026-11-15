use super::*;

mod protein_data_tests {
  use super::*;

  #[test]
  fn names_and_sequences() {
    clear_state();
    assert_eq!(
      interpret(r#"ProteinData["CFL1", "Name"]"#).unwrap(),
      "cofilin 1"
    );
    assert_eq!(
      interpret(r#"ProteinData["CFL2", "StandardName"]"#).unwrap(),
      "CFL2"
    );
    assert_eq!(
      interpret(r#"StringTake[ProteinData["CFL2", "Sequence"], 12]"#).unwrap(),
      "MASGVTVNDEVI"
    );
    assert_eq!(
      interpret(r#"ProteinData[{"CFL1", "CFL2"}, "SequenceLength"]"#).unwrap(),
      "{166, 166}"
    );
    assert_eq!(
      interpret(
        r#"StringLength[ProteinData["CFL1", "Sequence"]] == ProteinData["CFL1", "SequenceLength"]"#
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn go_classes_and_their_names_line_up() {
    clear_state();
    assert_eq!(
      interpret(r#"ProteinData["CFL2", "MolecularFunctions"]"#).unwrap(),
      "{ActinFilamentBinding}"
    );
    assert_eq!(
      interpret(r#"ProteinData["CFL2", "MolecularFunctions", "Name"]"#)
        .unwrap(),
      "{actin filament binding}"
    );
    // Every class list and its "Name" list have the same length and the
    // class names are the camel-cased term names, position by position.
    assert_eq!(
      interpret(
        r#"And @@ Flatten[Table[
          MapThread[
            #1 == StringJoin[Capitalize /@ StringSplit[#2, Except[WordCharacter]]] &,
            {ProteinData[p, prop], ProteinData[p, prop, "Name"]}],
          {p, {"CFL1", "ACTN1", "MYH9"}},
          {prop, {"BiologicalProcesses", "CellularComponents", "MolecularFunctions"}}]]"#
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn classes_list_their_direct_members() {
    clear_state();
    assert_eq!(
      interpret(r#"MemberQ[ProteinData["Classes"], "ActinBinding"]"#).unwrap(),
      "True"
    );
    // A class holds exactly the proteins that carry its term.
    assert_eq!(
      interpret(
        r#"With[{m = ProteinData["ActinFilamentBinding"]},
          {MemberQ[m, "CFL1"], MemberQ[m, "CFL2"],
           AllTrue[m, MemberQ[Join[ProteinData[#, "MolecularFunctions"],
             ProteinData[#, "BiologicalProcesses"],
             ProteinData[#, "CellularComponents"]], "ActinFilamentBinding"] &]}]"#
      )
      .unwrap(),
      "{True, True, True}"
    );
    assert_eq!(
      interpret(r#"Length[ProteinData["ActinBinding"]] > 100"#).unwrap(),
      "True"
    );
    // Entities and classes come in canonical order.
    assert_eq!(
      interpret(r#"With[{l = ProteinData[]}, l === Sort[l]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"With[{l = ProteinData["Classes"]}, l === Sort[l]]"#)
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn unknown_entities_and_properties() {
    clear_state();
    assert_eq!(
      interpret(r#"ProteinData["NotAProtein"]"#).unwrap(),
      "ProteinData[NotAProtein]"
    );
    assert_eq!(
      interpret(r#"ProteinData["CFL1", "NoSuchProperty"]"#).unwrap(),
      "Missing[NotAvailable]"
    );
  }

  // Aligning two bundled sequences and drawing the alignment wheel works
  // end to end (the pattern of a Manipulate over protein pairs).
  #[test]
  fn alignment_of_bundled_sequences() {
    clear_state();
    assert_eq!(
      interpret(
        r#"Take[SequenceAlignment[ProteinData["CFL1", "Sequence"],
          ProteinData["CFL2", "Sequence"], MergeDifferences -> False], 4]"#
      )
      .unwrap(),
      "{MASGV, {A, T}, V, {S, N}}"
    );
  }
}
