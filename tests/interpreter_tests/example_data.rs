use super::*;

mod example_data_tests {
  use super::*;

  /// Every bundled network, with the vertex and edge counts of the
  /// published dataset.
  const NETWORKS: &[(&str, usize, usize)] = &[
    ("ZacharysKarateClub", 34, 78),
    ("DolphinSocialNetwork", 62, 159),
    ("LesMiserables", 77, 254),
    ("BooksAboutUSPolitics", 105, 441),
    ("WordAdjacencies", 112, 425),
  ];

  #[test]
  fn lists_the_bundled_types() {
    clear_state();
    assert_eq!(interpret("ExampleData[]").unwrap(), "{NetworkGraph}");
  }

  #[test]
  fn lists_the_entries_of_a_type_as_type_name_pairs() {
    clear_state();
    assert_eq!(
      interpret("ExampleData[\"NetworkGraph\"][[1]]").unwrap(),
      "{NetworkGraph, ZacharysKarateClub}"
    );
    assert_eq!(
      interpret("Length[ExampleData[\"NetworkGraph\"]]").unwrap(),
      NETWORKS.len().to_string()
    );
    assert_eq!(
      interpret("Union[First /@ ExampleData[\"NetworkGraph\"]]").unwrap(),
      "{NetworkGraph}"
    );
  }

  #[test]
  fn every_network_has_its_published_size() {
    clear_state();
    for (name, vertices, edges) in NETWORKS {
      let graph = format!("ExampleData[{{\"NetworkGraph\", \"{name}\"}}]");
      assert_eq!(
        interpret(&format!("{{VertexCount[{graph}], EdgeCount[{graph}]}}"))
          .unwrap(),
        format!("{{{vertices}, {edges}}}"),
        "{name}"
      );
    }
  }

  #[test]
  fn a_network_evaluates_to_a_graph() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}]]"
      )
      .unwrap(),
      "Graph"
    );
    // Zachary's members are numbered; the other datasets name their nodes.
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}, \
         \"VertexList\"][[1 ;; 3]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"LesMiserables\"}, \
         \"VertexList\"][[1 ;; 2]]"
      )
      .unwrap(),
      "{Myriel, Napoleon}"
    );
  }

  #[test]
  fn properties_agree_with_the_graph() {
    clear_state();
    const G: &str = "ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}]";
    const S: &str = "{\"NetworkGraph\", \"ZacharysKarateClub\"}";
    assert_eq!(
      interpret(&format!("ExampleData[{S}, \"VertexCount\"]")).unwrap(),
      "34"
    );
    assert_eq!(
      interpret(&format!("ExampleData[{S}, \"EdgeCount\"]")).unwrap(),
      "78"
    );
    assert_eq!(
      interpret(&format!("ExampleData[{S}, \"Name\"]")).unwrap(),
      "ZacharysKarateClub"
    );
    assert_eq!(
      interpret(&format!(
        "ExampleData[{S}, \"VertexList\"] === VertexList[{G}]"
      ))
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!("Length[ExampleData[{S}, \"EdgeRules\"]]")).unwrap(),
      "78"
    );
    // The adjacency matrix is symmetric with 2 × 78 ones.
    assert_eq!(
      interpret(&format!(
        "Total[Flatten[ExampleData[{S}, \"AdjacencyMatrix\"]]]"
      ))
      .unwrap(),
      "156"
    );
    assert_eq!(
      interpret(&format!(
        "ExampleData[{S}, \"AdjacencyMatrix\"] === \
         Transpose[ExampleData[{S}, \"AdjacencyMatrix\"]]"
      ))
      .unwrap(),
      "True"
    );
    assert!(
      interpret(&format!("ExampleData[{S}, \"Source\"]"))
        .unwrap()
        .contains("Zachary")
    );
    assert!(
      interpret(&format!("ExampleData[{S}, \"Description\"]"))
        .unwrap()
        .contains("karate club")
    );
  }

  #[test]
  fn a_network_can_be_drawn() {
    clear_state();
    let svg = interpret(
      "ExportString[ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}], \
       \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(svg.matches("<ellipse").count(), 34);
  }

  #[test]
  fn unknown_types_and_names_stay_unevaluated() {
    clear_state();
    // Nothing is guessed: a dataset Woxi does not bundle comes back
    // unevaluated rather than as wrong data.
    assert_eq!(
      interpret("ExampleData[{\"NetworkGraph\", \"NoSuchNetwork\"}]").unwrap(),
      "ExampleData[{NetworkGraph, NoSuchNetwork}]"
    );
    assert_eq!(
      interpret("ExampleData[\"Text\"]").unwrap(),
      "ExampleData[Text]"
    );
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}, \"Nope\"]"
      )
      .unwrap(),
      "ExampleData[{NetworkGraph, ZacharysKarateClub}, Nope]"
    );
  }
}
