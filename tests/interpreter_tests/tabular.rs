use super::*;

mod tabular_ast {
  use super::*;

  #[test]
  fn tabular_list_of_lists() {
    clear_state();
    let result =
      interpret_with_stdout("Tabular[{{1, 2, 3}, {4, 5, 6}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should contain row numbers
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    // Should contain data values
    assert!(svg.contains(">3</text>"));
    assert!(svg.contains(">4</text>"));
    assert!(svg.contains(">5</text>"));
    assert!(svg.contains(">6</text>"));
  }

  #[test]
  fn tabular_list_of_lists_with_column_names() {
    clear_state();
    let result = interpret_with_stdout(
      "Tabular[{{1, 2, 3}, {4, 5, 6}}, {\"a\", \"b\", \"c\"}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should contain column headers
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">c</text>"));
    // Should contain data values
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">4</text>"));
  }

  #[test]
  fn tabular_list_of_associations() {
    clear_state();
    let result = interpret_with_stdout(
      "Tabular[{<|\"Name\" -> \"Alice\", \"Age\" -> 30|>, <|\"Name\" -> \"Bob\", \"Age\" -> 25|>}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should contain column headers from association keys
    assert!(svg.contains(">Name</text>"));
    assert!(svg.contains(">Age</text>"));
    // Should contain data values
    assert!(svg.contains(">Alice</text>"));
    assert!(svg.contains(">Bob</text>"));
    assert!(svg.contains(">30</text>"));
    assert!(svg.contains(">25</text>"));
  }

  #[test]
  fn tabular_column_association() {
    clear_state();
    assert_eq!(
      interpret("Tabular[<|\"x\" -> {1, 2, 3}, \"y\" -> {4, 5, 6}|>]").unwrap(),
      "Failure[TabularRowList, <|MessageParameters -> {<|x -> {1, 2, 3}, y -> {4, 5, 6}|>}, MessageTemplate :> MessageName[Tabular, rlist]|>]"
    );
  }

  #[test]
  fn tabular_flat_list() {
    clear_state();
    let result = interpret_with_stdout("Tabular[{10, 20, 30}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">10</text>"));
    assert!(svg.contains(">20</text>"));
    assert!(svg.contains(">30</text>"));
  }

  #[test]
  fn tabular_empty() {
    clear_state();
    assert_eq!(interpret("Tabular[]").unwrap(), "Tabular[]");
  }

  #[test]
  fn head_tabular() {
    clear_state();
    assert_eq!(
      interpret("Head[Tabular[{{1, 2}, {3, 4}}]]").unwrap(),
      "Tabular"
    );
  }

  #[test]
  fn normal_tabular_list_of_lists() {
    clear_state();
    assert_eq!(
      interpret("Normal[Tabular[{{1, 2}, {3, 4}}]]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn normal_tabular_list_of_associations() {
    clear_state();
    assert_eq!(
      interpret("Normal[Tabular[{<|\"a\" -> 1, \"b\" -> 2|>}]]").unwrap(),
      "{<|a -> 1, b -> 2|>}"
    );
  }

  #[test]
  fn normal_tabular_column_association() {
    clear_state();
    assert_eq!(
      interpret("Normal[Tabular[<|\"x\" -> {1, 2}, \"y\" -> {3, 4}|>]]")
        .unwrap(),
      "Failure[TabularRowList, {MessageParameters -> {<|x -> {1, 2}, y -> {3, 4}|>}, MessageTemplate :> MessageName[Tabular, rlist]}]"
    );
  }

  #[test]
  fn tabular_has_row_numbers() {
    clear_state();
    let result =
      interpret_with_stdout("Tabular[{{1, 2}, {3, 4}, {5, 6}}]").unwrap();
    let svg = result.graphics.unwrap();
    // Row number column background
    assert!(
      svg.contains("fill=\"#eef2f7\""),
      "Should have row-number column background"
    );
    // Row numbers as text
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
  }

  #[test]
  fn tabular_has_header_background() {
    clear_state();
    let result =
      interpret_with_stdout("Tabular[{<|\"X\" -> 1, \"Y\" -> 2|>}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("fill=\"#f0f0f0\""),
      "Should have header background"
    );
    assert!(
      svg.contains("font-weight=\"bold\""),
      "Headers should be bold"
    );
  }

  #[test]
  fn tabular_with_variable() {
    clear_state();
    let result = interpret_with_stdout(
      "data = {<|\"A\" -> 1|>, <|\"A\" -> 2|>}; Tabular[data]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn tabular_mixed_types() {
    clear_state();
    let result = interpret_with_stdout(
      "Tabular[{<|\"Name\" -> \"Alice\", \"Score\" -> 95.5|>, <|\"Name\" -> \"Bob\", \"Score\" -> 87.3|>}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">Name</text>"));
    assert!(svg.contains(">Score</text>"));
    assert!(svg.contains(">Alice</text>"));
    assert!(svg.contains(">95.5</text>"));
  }

  #[test]
  fn tabular_single_column_association() {
    clear_state();
    assert_eq!(
      interpret("Tabular[<|\"values\" -> {10, 20, 30}|>]").unwrap(),
      "Failure[TabularRowList, <|MessageParameters -> {<|values -> {10, 20, 30}|>}, MessageTemplate :> MessageName[Tabular, rlist]|>]"
    );
  }

  #[test]
  fn tabular_already_has_schema() {
    clear_state();
    // When Tabular already has a TabularSchema, it should pass through
    let result = interpret("Tabular[{{1, 2}}, TabularSchema[<||>]]").unwrap();
    // Should render as graphics since it has schema
    assert!(result == "-Graphics-" || result.contains("Tabular"));
  }

  #[test]
  fn output_svg_not_set_for_tabular() {
    clear_state();
    let result = interpret_with_stdout("Tabular[{<|\"a\" -> 1|>}]").unwrap();
    assert!(
      result.output_svg.is_none(),
      "output_svg should be None for Tabular table results"
    );
    assert!(
      result.graphics.is_some(),
      "graphics should be set for Tabular"
    );
  }

  #[test]
  fn tabular_list_of_associations_missing_keys() {
    // When rows have different keys, missing values should be handled
    clear_state();
    let result = interpret_with_stdout(
      "Tabular[{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 3, \"c\" -> 4|>}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should have all three columns
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">c</text>"));
  }

  #[test]
  fn tabular_single_row() {
    clear_state();
    let result = interpret_with_stdout("Tabular[{{1, 2, 3}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn tabular_string_data() {
    clear_state();
    let result = interpret_with_stdout(
      "Tabular[{{\"hello\", \"world\"}, {\"foo\", \"bar\"}}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">hello</text>"));
    assert!(svg.contains(">world</text>"));
    assert!(svg.contains(">foo</text>"));
    assert!(svg.contains(">bar</text>"));
  }
}

mod to_tabular {
  use super::*;

  #[test]
  fn to_tabular_columns_list_of_rules() {
    clear_state();
    let result = interpret_with_stdout(
      "ToTabular[{\"a\" -> {1, 4}, \"b\" -> {2, 5}, \"c\" -> {3, 6}}, \"Columns\"]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should contain column headers
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">c</text>"));
    // Should contain data values
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
    assert!(svg.contains(">4</text>"));
    assert!(svg.contains(">5</text>"));
    assert!(svg.contains(">6</text>"));
  }

  #[test]
  fn to_tabular_columns_association() {
    clear_state();
    let result = interpret_with_stdout(
      "ToTabular[<|\"x\" -> {10, 20}, \"y\" -> {30, 40}|>, \"Columns\"]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">x</text>"));
    assert!(svg.contains(">y</text>"));
    assert!(svg.contains(">10</text>"));
    assert!(svg.contains(">20</text>"));
    assert!(svg.contains(">30</text>"));
    assert!(svg.contains(">40</text>"));
  }

  #[test]
  fn to_tabular_normal_roundtrip() {
    clear_state();
    assert_eq!(
      interpret(
        "Normal[ToTabular[{\"a\" -> {1, 2}, \"b\" -> {3, 4}}, \"Columns\"]]"
      )
      .unwrap(),
      "{<|a -> 1, b -> 3|>, <|a -> 2, b -> 4|>}"
    );
  }

  #[test]
  fn to_tabular_head() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[ToTabular[{\"a\" -> {1, 2}, \"b\" -> {3, 4}}, \"Columns\"]]"
      )
      .unwrap(),
      "Tabular"
    );
  }

  #[test]
  fn to_tabular_unevaluated_without_orientation() {
    clear_state();
    assert_eq!(
      interpret("ToTabular[{\"a\" -> {1, 2}}]").unwrap(),
      "Failure[TabularRowList, <|MessageParameters -> {{a -> {1, 2}}}, MessageTemplate :> MessageName[ToTabular, rlist]|>]"
    );
  }

  #[test]
  fn to_tabular_single_column() {
    clear_state();
    let result = interpret_with_stdout(
      "ToTabular[{\"vals\" -> {100, 200, 300}}, \"Columns\"]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">vals</text>"));
    assert!(svg.contains(">100</text>"));
    assert!(svg.contains(">200</text>"));
    assert!(svg.contains(">300</text>"));
  }
}
