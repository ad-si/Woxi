use super::*;

mod dataset_ast {
  use super::*;

  #[test]
  fn dataset_single_association() {
    clear_state();
    let result = interpret_with_stdout(
      "Dataset[<|\"Name\" -> \"John\", \"Age\" -> 30, \"City\" -> \"NYC\"|>]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">Name</text>"));
    assert!(svg.contains(">Age</text>"));
    assert!(svg.contains(">City</text>"));
    assert!(svg.contains(">John</text>"));
    assert!(svg.contains(">30</text>"));
    assert!(svg.contains(">NYC</text>"));
  }

  #[test]
  fn dataset_list_of_associations() {
    clear_state();
    let result = interpret_with_stdout(
      "Dataset[{<|\"Name\" -> \"John\", \"Age\" -> 30|>, <|\"Name\" -> \"Jane\", \"Age\" -> 28|>}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">Name</text>"));
    assert!(svg.contains(">Age</text>"));
    assert!(svg.contains(">John</text>"));
    assert!(svg.contains(">Jane</text>"));
    assert!(svg.contains(">30</text>"));
    assert!(svg.contains(">28</text>"));
  }

  #[test]
  fn dataset_plain_list() {
    // Plain lists (not associations) don't render as table
    assert_eq!(
      interpret("Dataset[{1, 2, 3}]").unwrap(),
      "Dataset[{1, 2, 3}, TypeSystem`Vector[TypeSystem`Atom[Integer], 3], <||>]"
    );
  }

  #[test]
  fn dataset_atom() {
    assert_eq!(
      interpret("Dataset[42]").unwrap(),
      "Dataset[42, TypeSystem`Atom[Integer], <||>]"
    );
  }

  #[test]
  fn dataset_string_list() {
    assert_eq!(
      interpret("Dataset[{\"a\", \"b\"}]").unwrap(),
      "Dataset[{a, b}, TypeSystem`Vector[TypeSystem`Atom[String], 2], <||>]"
    );
  }

  #[test]
  fn dataset_real_list() {
    assert_eq!(
      interpret("Dataset[{1.5, 2.3}]").unwrap(),
      "Dataset[{1.5, 2.3}, TypeSystem`Vector[TypeSystem`Atom[Real], 2], <||>]"
    );
  }

  #[test]
  fn dataset_mixed_types_tuple() {
    assert_eq!(
      interpret("Dataset[{1, \"a\"}]").unwrap(),
      "Dataset[{1, a}, TypeSystem`Tuple[{TypeSystem`Atom[Integer], TypeSystem`Atom[String]}], <||>]"
    );
  }

  #[test]
  fn dataset_boolean_list() {
    assert_eq!(
      interpret("Dataset[{True, False}]").unwrap(),
      "Dataset[{True, False}, TypeSystem`Vector[TypeSystem`Atom[TypeSystem`Boolean], 2], <||>]"
    );
  }

  #[test]
  fn dataset_nested_lists() {
    assert_eq!(
      interpret("Dataset[{{1, 2}, {3, 4}}]").unwrap(),
      "Dataset[{{1, 2}, {3, 4}}, TypeSystem`Vector[TypeSystem`Vector[TypeSystem`Atom[Integer], 2], 2], <||>]"
    );
  }

  #[test]
  fn dataset_homogeneous_assoc() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
  }

  #[test]
  fn dataset_assoc_with_list_values() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|\"a\" -> {1, 2}, \"b\" -> {3, 4}|>]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
  }

  #[test]
  fn dataset_nested_associations() {
    clear_state();
    let result = interpret_with_stdout(
      "Dataset[<|\"a\" -> <|\"x\" -> 1|>, \"b\" -> <|\"x\" -> 2|>|>]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
  }

  #[test]
  fn dataset_with_variable() {
    clear_state();
    let result = interpret_with_stdout(
      "assoc = <|\"Name\" -> \"John\", \"Age\" -> 30, \"City\" -> \"NYC\"|>; Dataset[assoc]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn dataset_list_variable() {
    clear_state();
    let result = interpret_with_stdout(
      "data = {<|\"Name\" -> \"John\", \"Age\" -> 30|>, <|\"Name\" -> \"Jane\", \"Age\" -> 28|>}; Dataset[data]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn normal_dataset() {
    assert_eq!(
      interpret("Normal[Dataset[<|\"Name\" -> \"John\", \"Age\" -> 30|>]]")
        .unwrap(),
      "<|Name -> John, Age -> 30|>"
    );
  }

  #[test]
  fn head_dataset() {
    assert_eq!(
      interpret("Head[Dataset[<|\"Name\" -> \"John\"|>]]").unwrap(),
      "Dataset"
    );
  }

  #[test]
  fn dataset_integer_keys() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|1 -> \"x\", 2 -> \"y\"|>]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">x</text>"));
    assert!(svg.contains(">y</text>"));
  }

  #[test]
  fn dataset_mixed_assoc_values() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|\"a\" -> 1, \"b\" -> \"x\"|>]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">x</text>"));
  }

  #[test]
  fn dataset_already_typed() {
    // Dataset with 3 args should pass through
    assert_eq!(
      interpret("Dataset[{1, 2}, foo, bar]").unwrap(),
      "Dataset[{1, 2}, foo, bar]"
    );
  }

  #[test]
  fn dataset_assoc_keys_are_bold() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|\"Name\" -> \"John\", \"Age\" -> 30|>]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("font-weight=\"bold\""),
      "Key column should use bold font"
    );
  }

  #[test]
  fn dataset_assoc_has_key_column_background() {
    clear_state();
    let result =
      interpret_with_stdout("Dataset[<|\"Name\" -> \"John\"|>]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<rect"),
      "Should have a background rect for the key column"
    );
  }

  #[test]
  fn dataset_list_has_header_background() {
    clear_state();
    let result = interpret_with_stdout("Dataset[{<|\"X\" -> 1|>}]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains("<rect"),
      "Should have a background rect for the header row"
    );
  }

  #[test]
  fn output_svg_not_set_for_dataset() {
    clear_state();
    let result = interpret_with_stdout("Dataset[<|\"a\" -> 1|>]").unwrap();
    assert!(
      result.output_svg.is_none(),
      "output_svg should be None for Dataset table results"
    );
    assert!(
      result.graphics.is_some(),
      "graphics should be set for Dataset"
    );
  }
}
