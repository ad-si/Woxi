use super::*;

mod example_data_tests {
  use super::*;

  /// Every bundled network, with the vertex and edge counts of the
  /// published dataset.
  const NETWORKS: &[(&str, usize, usize)] = &[
    ("ZacharyKarateClub", 34, 78),
    ("DolphinSocialNetwork", 62, 159),
    ("LesMiserables", 77, 254),
    ("USPoliticsBooks", 105, 441),
    ("WordAdjacencies", 112, 425),
  ];

  /// Which datasets an implementation ships is its own business — Wolfram's
  /// catalogue is far larger than the one Woxi bundles — so the catalogue
  /// tests assert the shape of the answer and the presence of what is
  /// bundled, never the catalogue itself.
  #[test]
  fn lists_the_bundled_types() {
    clear_state();
    assert_eq!(
      interpret("MemberQ[ExampleData[], \"NetworkGraph\"]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("And @@ (StringQ /@ ExampleData[])").unwrap(),
      "True"
    );
  }

  #[test]
  fn lists_the_entries_of_a_type_as_type_name_pairs() {
    clear_state();
    assert_eq!(
      interpret("Length[ExampleData[\"NetworkGraph\"]] > 0").unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "And @@ (MatchQ[#, {_String, _String}] & /@ \
         ExampleData[\"NetworkGraph\"])"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Union[First /@ ExampleData[\"NetworkGraph\"]]").unwrap(),
      "{NetworkGraph}"
    );
    // Everything Woxi bundles is listed, and nothing else is claimed.
    assert_eq!(
      interpret("Length[ExampleData[\"NetworkGraph\"]]").unwrap(),
      NETWORKS.len().to_string()
    );
    for (name, _, _) in NETWORKS {
      assert_eq!(
        interpret(&format!(
          "MemberQ[ExampleData[\"NetworkGraph\"], \
           {{\"NetworkGraph\", \"{name}\"}}]"
        ))
        .unwrap(),
        "True",
        "{name}"
      );
    }
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
      interpret("Head[ExampleData[{\"NetworkGraph\", \"ZacharyKarateClub\"}]]")
        .unwrap(),
      "Graph"
    );
    // Zachary's members are numbered; the other datasets name their nodes.
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"ZacharyKarateClub\"}, \
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
    const G: &str = "ExampleData[{\"NetworkGraph\", \"ZacharyKarateClub\"}]";
    const S: &str = "{\"NetworkGraph\", \"ZacharyKarateClub\"}";
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
      "ZacharyKarateClub"
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
      "ExportString[ExampleData[{\"NetworkGraph\", \"ZacharyKarateClub\"}], \
       \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(svg.matches("<ellipse").count(), 34);
  }

  /// A dataset that Wolfram's catalogue also answers to under a second
  /// spelling answers to it here: `"ZacharysKarateClub"` and
  /// `"BooksAboutUSPolitics"` reach the same networks as the catalogue
  /// names, so a script written against either spelling runs.
  #[test]
  fn alternate_dataset_spellings_resolve() {
    clear_state();
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}, \
         \"VertexCount\"]"
      )
      .unwrap(),
      "34"
    );
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"BooksAboutUSPolitics\"}, \
         \"VertexCount\"]"
      )
      .unwrap(),
      "105"
    );
    // The catalogue itself lists each network once, under Wolfram's name.
    assert_eq!(
      interpret(
        "Count[ExampleData[\"NetworkGraph\"], \
         {_, \"ZacharyKarateClub\" | \"ZacharysKarateClub\"}]"
      )
      .unwrap(),
      "1"
    );
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
      interpret("ExampleData[\"NoSuchType\"]").unwrap(),
      "ExampleData[NoSuchType]"
    );
    assert_eq!(
      interpret(
        "ExampleData[{\"NetworkGraph\", \"ZacharyKarateClub\"}, \"Nope\"]"
      )
      .unwrap(),
      "ExampleData[{NetworkGraph, ZacharyKarateClub}, Nope]"
    );
  }

  /// Woxi bundles no photographic data, only the `"TestImage"` name
  /// catalogue — enough for a script to build UI (a `Control` popup,
  /// `Thread[...]` over the name list) from `ExampleData["TestImage"]`
  /// without the actual pixels being available.
  #[test]
  fn lists_the_test_image_catalogue() {
    clear_state();
    assert_eq!(
      interpret("MemberQ[ExampleData[], \"TestImage\"]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Length[ExampleData[\"TestImage\"]] > 0").unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "And @@ (MatchQ[#, {_String, _String}] & /@ \
         ExampleData[\"TestImage\"])"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Union[First /@ ExampleData[\"TestImage\"]]").unwrap(),
      "{TestImage}"
    );
    // The catalogue lists each name once.
    assert_eq!(
      interpret(
        "Length[ExampleData[\"TestImage\"]] == \
         Length[Union[Last /@ ExampleData[\"TestImage\"]]]"
      )
      .unwrap(),
      "True"
    );
    for name in ["Couple", "Mandrill", "House", "Peppers"] {
      assert_eq!(
        interpret(&format!(
          "MemberQ[ExampleData[\"TestImage\"], {{\"TestImage\", \"{name}\"}}]"
        ))
        .unwrap(),
        "True",
        "{name}"
      );
    }
  }

  /// The name catalogue is bundled, but no pixel data is: asking for one
  /// of the catalogued images stays unevaluated rather than returning
  /// invented pixels.
  #[test]
  fn test_image_pixel_data_stays_unbundled() {
    clear_state();
    assert_eq!(
      interpret("ExampleData[{\"TestImage\", \"Couple\"}]").unwrap(),
      "ExampleData[{TestImage, Couple}]"
    );
  }

  /// The Wolfram Demonstration "Histogram Equalization" builds its image
  /// picker from exactly this pattern:
  /// `imageNames = ExampleData["TestImage"]; Thread[imageNames ->
  /// imageNames[[All, 2]]]` — a rule list pairing each `{"TestImage",
  /// name}` entry with its bare name. This regression test guards that
  /// the catalogue is a concrete, evaluated list so `Thread` (and so the
  /// `Control` popup built from it) actually has something to work with.
  #[test]
  fn thread_over_the_catalogue_builds_display_rules() {
    clear_state();
    assert_eq!(
      interpret(
        "imageNames = ExampleData[\"TestImage\"]; \
         MatchQ[Thread[imageNames -> imageNames[[All, 2]]], \
         {({_String, _String} -> _String) ..}]"
      )
      .unwrap(),
      "True"
    );
  }
}
