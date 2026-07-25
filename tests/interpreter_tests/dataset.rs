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
    clear_state();
    let result = interpret_with_stdout("Dataset[{1, 2, 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
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
    clear_state();
    let result = interpret_with_stdout("Dataset[{\"a\", \"b\"}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
  }

  #[test]
  fn dataset_real_list() {
    clear_state();
    let result = interpret_with_stdout("Dataset[{1.5, 2.3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1.5</text>"));
    assert!(svg.contains(">2.3</text>"));
  }

  #[test]
  fn dataset_mixed_types_tuple() {
    clear_state();
    let result = interpret_with_stdout("Dataset[{1, \"a\"}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">a</text>"));
  }

  #[test]
  fn dataset_boolean_list() {
    clear_state();
    let result = interpret_with_stdout("Dataset[{True, False}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">True</text>"));
    assert!(svg.contains(">False</text>"));
  }

  #[test]
  fn dataset_nested_lists() {
    clear_state();
    let result = interpret_with_stdout("Dataset[{{1, 2}, {3, 4}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>") || svg.contains(">{1, 2}</text>"));
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
    // Dataset with 3 args containing list data renders as graphics
    clear_state();
    let result = interpret_with_stdout("Dataset[{1, 2}, foo, bar]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
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

  #[test]
  fn dataset_all_column() {
    clear_state();
    let result = interpret_with_stdout(
      "ds = Dataset[{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 3, \"b\" -> 4|>}]; ds[All, \"a\"]"
    ).unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">3</text>"));
  }

  #[test]
  fn dataset_all_column_mixed_types() {
    clear_state();
    let result = interpret_with_stdout(
      "ds = Dataset[{<|\"x\" -> 1, \"y\" -> \"hello\"|>, <|\"x\" -> 2, \"y\" -> \"world\"|>}]; ds[All, \"y\"]"
    ).unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">hello</text>"));
    assert!(svg.contains(">world</text>"));
  }

  #[test]
  fn dataset_all_column_boolean() {
    clear_state();
    let result = interpret_with_stdout(
      "ds = Dataset[{<|\"s\" -> True|>, <|\"s\" -> False|>}]; ds[All, \"s\"]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">True</text>"));
    assert!(svg.contains(">False</text>"));
  }

  #[test]
  fn delete_missing_dataset() {
    clear_state();
    let result = interpret_with_stdout(
      "ds = Dataset[{<|\"a\" -> 1|>, <|\"a\" -> 2|>}]; DeleteMissing[ds[All, \"a\"]]"
    ).unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
  }

  #[test]
  fn dataset_titanic_example() {
    clear_state();
    let result = interpret_with_stdout(
      "titanic = Dataset[{\
        <|\"class\" -> \"1st\", \"age\" -> 29, \"sex\" -> \"female\", \"survived\" -> True|>,\
        <|\"class\" -> \"1st\", \"age\" -> 1, \"sex\" -> \"male\", \"survived\" -> True|>,\
        <|\"class\" -> \"1st\", \"age\" -> 2, \"sex\" -> \"female\", \"survived\" -> False|>,\
        <|\"class\" -> \"1st\", \"age\" -> 30, \"sex\" -> \"male\", \"survived\" -> False|>,\
        <|\"class\" -> \"1st\", \"age\" -> 25, \"sex\" -> \"female\", \"survived\" -> False|>\
      }]; titanic[All, \"age\"]"
    ).unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">29</text>"));
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">30</text>"));
    assert!(svg.contains(">25</text>"));
  }

  // Dataset[data][agg, "column"] applies a scalar aggregator over one column
  // and returns the bare result. Values verified against wolframscript.
  #[test]
  fn dataset_column_total() {
    clear_state();
    assert_eq!(
      interpret(
        "Dataset[{<|\"a\" -> 1, \"b\" -> 10|>, <|\"a\" -> 2, \"b\" -> 20|>, \
         <|\"a\" -> 3, \"b\" -> 30|>}][Total, \"a\"]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn dataset_column_mean() {
    clear_state();
    assert_eq!(
      interpret(
        "Dataset[{<|\"a\" -> 1, \"b\" -> 10|>, <|\"a\" -> 2, \"b\" -> 20|>, \
         <|\"a\" -> 3, \"b\" -> 30|>}][Mean, \"b\"]"
      )
      .unwrap(),
      "20"
    );
  }

  #[test]
  fn dataset_column_max_min() {
    clear_state();
    assert_eq!(
      interpret(
        "Dataset[{<|\"a\" -> 1|>, <|\"a\" -> 2|>, <|\"a\" -> 3|>}][Max, \"a\"]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "Dataset[{<|\"a\" -> 1|>, <|\"a\" -> 2|>, <|\"a\" -> 3|>}][Min, \"a\"]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn dataset_column_median() {
    clear_state();
    assert_eq!(
      interpret(
        "Dataset[{<|\"a\" -> 1|>, <|\"a\" -> 2|>, <|\"a\" -> 3|>}][Median, \"a\"]"
      )
      .unwrap(),
      "2"
    );
  }

  // Dataset[flatlist][f] applies f to the data; a scalar result unwraps.
  #[test]
  fn dataset_apply_scalar_aggregator() {
    clear_state();
    assert_eq!(interpret("Dataset[{1, 2, 3}][Mean]").unwrap(), "2");
    assert_eq!(interpret("Dataset[{1, 2, 3}][Total]").unwrap(), "6");
    assert_eq!(interpret("Dataset[{3, 1, 2}][Max]").unwrap(), "3");
    assert_eq!(interpret("Dataset[{3, 1, 2}][Min]").unwrap(), "1");
    assert_eq!(interpret("Dataset[{1, 2, 3, 4}][Length]").unwrap(), "4");
  }

  // A list-valued result stays a Dataset (rendered as -Graphics-); its
  // underlying data is the transformed list.
  #[test]
  fn dataset_apply_list_transform_rewraps() {
    clear_state();
    assert_eq!(
      interpret("Head[Dataset[{3, 1, 2}][Sort]]").unwrap(),
      "Dataset"
    );
    assert_eq!(
      interpret("Normal[Dataset[{3, 1, 2}][Sort]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("Normal[Dataset[{1, 2, 3, 4}][Reverse]]").unwrap(),
      "{4, 3, 2, 1}"
    );
  }

  // A dataset query is the same successive-level operator spec Query takes.
  // All values verified against wolframscript.
  const ROWS: &str = "ds = Dataset[{<|\"a\" -> 1, \"b\" -> \"x\"|>, \
                      <|\"a\" -> 2, \"b\" -> \"y\"|>, \
                      <|\"a\" -> 3, \"b\" -> \"z\"|>}]; ";

  #[test]
  fn dataset_filter_operator_rewraps() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Head[ds[Select[#a > 1 &]]]")).unwrap(),
      "Dataset"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[Select[#a > 1 &]]]")).unwrap(),
      "{<|a -> 2, b -> y|>, <|a -> 3, b -> z|>}"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[SortBy[-#a &], \"a\"]]")).unwrap(),
      "{3, 2, 1}"
    );
  }

  #[test]
  fn dataset_row_index() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[2]]")).unwrap(),
      "<|a -> 2, b -> y|>"
    );
    assert_eq!(interpret(&format!("{ROWS}Head[ds[2]]")).unwrap(), "Dataset");
    // A deeper key spec reaches an atom, which unwraps.
    assert_eq!(interpret(&format!("{ROWS}ds[2, \"a\"]")).unwrap(), "2");
  }

  #[test]
  fn dataset_row_list_spec() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[{{1, 3}}]]")).unwrap(),
      "{<|a -> 1, b -> x|>, <|a -> 3, b -> z|>}"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[{{1, 3}}, \"a\"]]")).unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn dataset_aggregator_returning_an_association() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[Counts, \"b\"]]")).unwrap(),
      "<|x -> 1, y -> 1, z -> 1|>"
    );
  }

  #[test]
  fn dataset_applies_arbitrary_functions_on_the_way_up() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Normal[ds[All, \"a\", f]]")).unwrap(),
      "{f[1], f[2], f[3]}"
    );
    assert_eq!(
      interpret("Normal[Dataset[{1, 2, 3}][f]]").unwrap(),
      "f[{1, 2, 3}]"
    );
  }

  // A key spec addresses an association; on an association-valued dataset it
  // looks the key up, on a list of rows Wolfram reports it as not applicable.
  #[test]
  fn dataset_key_spec() {
    clear_state();
    assert_eq!(
      interpret("Dataset[<|\"a\" -> 1, \"b\" -> 2|>][\"a\"]").unwrap(),
      "1"
    );
    assert!(
      interpret(&format!("{ROWS}ds[\"a\"]"))
        .unwrap()
        .starts_with("Dataset["),
      "a key spec over rows should stay unevaluated"
    );
  }

  #[test]
  fn keys_and_values_of_a_dataset() {
    clear_state();
    assert_eq!(
      interpret(&format!("{ROWS}Normal[Keys[ds]]")).unwrap(),
      "{{a, b}, {a, b}, {a, b}}"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[Values[ds]]")).unwrap(),
      "{{1, x}, {2, y}, {3, z}}"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[Keys[ds[1]]]")).unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret(&format!("{ROWS}Normal[Values[ds[1]]]")).unwrap(),
      "{1, x}"
    );
    assert_eq!(
      interpret("Normal[Keys[Dataset[<|\"a\" -> 1, \"b\" -> 2|>]]]").unwrap(),
      "{a, b}"
    );
  }
}
