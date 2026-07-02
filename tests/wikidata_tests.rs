use woxi::interpret;

mod wikidata_tests {
  use super::*;

  mod external_identifier_tests {
    use super::*;

    #[test]
    fn stays_symbolic() {
      assert_eq!(
        interpret("ExternalIdentifier[\"WikidataID\", \"Q405\"]").unwrap(),
        "ExternalIdentifier[WikidataID, Q405]"
      );
    }

    #[test]
    fn keeps_metadata_association() {
      assert_eq!(
        interpret(
          "ExternalIdentifier[\"WikidataID\", \"Q405\", <|\"Label\" -> \"Moon\"|>]"
        )
        .unwrap(),
        "ExternalIdentifier[WikidataID, Q405, <|Label -> Moon|>]"
      );
    }

    #[test]
    fn assignment_round_trips() {
      assert_eq!(
        interpret(
          "moon = ExternalIdentifier[\"WikidataID\", \"Q405\", <|\"Label\" -> \"Moon\"|>]; moon"
        )
        .unwrap(),
        "ExternalIdentifier[WikidataID, Q405, <|Label -> Moon|>]"
      );
    }
  }

  mod wikidata_data_tests {
    use super::*;

    #[test]
    fn invalid_arguments_stay_symbolic() {
      assert_eq!(
        interpret("WikidataData[1, 2]").unwrap(),
        "WikidataData[1, 2]"
      );
    }

    #[test]
    fn wrong_identifier_kind_stays_symbolic() {
      // Items must be Q-ids and properties P-ids.
      assert_eq!(
        interpret("WikidataData[\"P2067\", \"Q405\"]").unwrap(),
        "WikidataData[P2067, Q405]"
      );
    }

    #[test]
    fn wrong_argument_count_stays_symbolic() {
      assert_eq!(
        interpret("WikidataData[\"Q405\"]").unwrap(),
        "WikidataData[Q405]"
      );
    }

    // The remaining tests hit the live wikidata.org API and are therefore
    // excluded from the default offline suite; run them on demand with
    // `cargo test -- --ignored` (or `make test-slow`).

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn moon_mass_example() {
      assert_eq!(
        interpret(
          "moon = ExternalIdentifier[\"WikidataID\", \"Q405\", <|\"Label\" -> \"Moon\"|>]; \
           mass = ExternalIdentifier[\"WikidataID\", \"P2067\", <|\"Label\" -> \"mass\"|>]; \
           WikidataData[moon, mass] // First"
        )
        .unwrap(),
        "Quantity[73.4767, Yottagrams]"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn flag_of_germany_image_url_example() {
      assert_eq!(
        interpret(
          "flagOfGermany = ExternalIdentifier[\"WikidataID\", \"Q48160\", <|\"Label\" -> \"Flag of Germany\"|>]; \
           image = ExternalIdentifier[\"WikidataID\", \"P18\", <|\"Label\" -> \"image\"|>]; \
           WikidataData[flagOfGermany, image] // First"
        )
        .unwrap(),
        "URL[http://commons.wikimedia.org/wiki/Special:FilePath/Flag%20of%20Germany.svg]"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn flag_of_germany_import_example() {
      assert_eq!(
        interpret(
          "flagOfGermany = ExternalIdentifier[\"WikidataID\", \"Q48160\", <|\"Label\" -> \"Flag of Germany\"|>]; \
           image = ExternalIdentifier[\"WikidataID\", \"P18\", <|\"Label\" -> \"image\"|>]; \
           WikidataData[flagOfGermany, image] // First // Import"
        )
        .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn raw_identifier_strings_and_lists() {
      assert_eq!(
        interpret("WikidataData[\"Q405\", \"P2067\"]").unwrap(),
        "{Quantity[73.4767, Yottagrams]}"
      );
      // An item list maps to one result list per item.
      assert_eq!(
        interpret("WikidataData[{\"Q405\", \"Q308\"}, \"P2067\"]").unwrap(),
        "{{Quantity[73.4767, Yottagrams]}, {Quantity[330, Yottagrams]}}"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn entity_values_become_external_identifiers() {
      assert_eq!(
        interpret("WikidataData[\"Q405\", \"P361\"] // First").unwrap(),
        "ExternalIdentifier[WikidataID, Q18589965, \
         <|Label -> Earth-Moon system, Description -> Moon orbiting Earth|>]"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn time_values_become_date_objects() {
      // Albert Einstein's date of birth.
      assert_eq!(
        interpret("WikidataData[\"Q937\", \"P569\"]").unwrap(),
        "{DateObject[{1879, 3, 14}, Day]}"
      );
    }

    #[test]
    #[ignore = "network: queries the live wikidata.org API"]
    fn no_values_give_empty_list() {
      // The Moon has no "image of grave" (P1442) statement.
      assert_eq!(
        interpret("WikidataData[\"Q405\", \"P1442\"]").unwrap(),
        "{}"
      );
    }
  }

  mod svg_import_tests {
    use super::*;

    #[test]
    fn local_svg_imports_as_graphics() {
      let path = std::env::temp_dir().join("woxi_import_test.svg");
      std::fs::write(
        &path,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">\
         <rect width=\"10\" height=\"10\" fill=\"red\"/></svg>",
      )
      .unwrap();
      assert_eq!(
        interpret(&format!("Import[\"{}\"]", path.display())).unwrap(),
        "-Graphics-"
      );
      std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_svg_file_fails_like_wolframscript() {
      assert_eq!(
        interpret("Import[\"/nonexistent/woxi_missing.svg\"]").unwrap(),
        "$Failed"
      );
    }
  }
}
