use super::*;

mod row_text_mode {
  use super::*;

  #[test]
  fn no_separator() {
    clear_state();
    assert_eq!(interpret("Row[{1, 2, 3}]").unwrap(), "123");
    assert_eq!(interpret("Row[{a, b, c}]").unwrap(), "abc");
  }

  #[test]
  fn with_string_separator() {
    clear_state();
    assert_eq!(interpret(r#"Row[{1, 2, 3}, ", "]"#).unwrap(), "1, 2, 3");
    assert_eq!(interpret(r#"Row[{a, b, c}, "+"]"#).unwrap(), "a+b+c");
  }

  #[test]
  fn with_spacer_separator() {
    clear_state();
    // Spacer[w] uses printer's points; ~7 pt per character.
    // Spacer[21] = round(21/7) = 3 spaces between items.
    assert_eq!(
      interpret(r#"Row[{"a", "b", "c"}, Spacer[21]]"#).unwrap(),
      "a   b   c"
    );
  }

  #[test]
  fn with_spacer_small() {
    clear_state();
    // Spacer[7] = round(7/7) = 1 space
    assert_eq!(
      interpret(r#"Row[{"a", "b", "c"}, Spacer[7]]"#).unwrap(),
      "a b c"
    );
  }

  #[test]
  fn with_spacer_14() {
    clear_state();
    // Spacer[14] = round(14/7) = 2 spaces
    assert_eq!(interpret(r#"Row[{"x", "y"}, Spacer[14]]"#).unwrap(), "x  y");
  }

  #[test]
  fn with_spacer_0() {
    clear_state();
    // Spacer[0] means no extra space
    assert_eq!(interpret(r#"Row[{"a", "b"}, Spacer[0]]"#).unwrap(), "ab");
  }

  #[test]
  fn with_spacer_list_form() {
    clear_state();
    // Spacer[{w, h}] — only the width (w) matters for Row separator
    // Spacer[{14, 20}] = round(14/7) = 2 spaces
    assert_eq!(
      interpret(r#"Row[{"a", "b"}, Spacer[{14, 20}]]"#).unwrap(),
      "a  b"
    );
  }

  #[test]
  fn with_spacer_list_form_3() {
    clear_state();
    // Spacer[{w, h, dh}]
    assert_eq!(
      interpret(r#"Row[{"a", "b"}, Spacer[{21, 10, 5}]]"#).unwrap(),
      "a   b"
    );
  }

  #[test]
  fn with_spacer_float() {
    clear_state();
    // Spacer[10.5] = round(10.5/7) = round(1.5) = 2 spaces
    assert_eq!(
      interpret(r#"Row[{"x", "y"}, Spacer[10.5]]"#).unwrap(),
      "x  y"
    );
  }

  #[test]
  fn spacer_single_element() {
    clear_state();
    assert_eq!(interpret(r#"Row[{"only"}, Spacer[10]]"#).unwrap(), "only");
  }

  #[test]
  fn spacer_empty_list() {
    clear_state();
    assert_eq!(interpret(r#"Row[{}, Spacer[10]]"#).unwrap(), "");
  }

  #[test]
  fn evaluates_arguments() {
    clear_state();
    assert_eq!(interpret("Row[{1 + 1, 2 + 2}]").unwrap(), "24");
    assert_eq!(interpret("Row[{1 + 1, 2 + 2}, Spacer[7]]").unwrap(), "2 4");
  }

  #[test]
  fn non_list_arg_stays_symbolic() {
    clear_state();
    assert_eq!(interpret("Row[x]").unwrap(), "Row[x]");
    assert_eq!(interpret("Row[5]").unwrap(), "Row[5]");
  }

  #[test]
  fn head() {
    clear_state();
    assert_eq!(interpret("Head[Row[{1, 2, 3}]]").unwrap(), "Row");
  }
}

mod row_visual_mode {
  use super::*;

  #[test]
  fn row_renders_svg() {
    clear_state();
    let result = interpret_with_stdout("Row[{1, 2, 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
  }

  #[test]
  fn row_renders_strings() {
    clear_state();
    let result = interpret_with_stdout(r#"Row[{"hello", "world"}]"#).unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">hello</text>"));
    assert!(svg.contains(">world</text>"));
  }

  #[test]
  fn row_with_spacer_renders_svg() {
    clear_state();
    let result = interpret_with_stdout(
      r#"Row[{"Item 1", "Item 2", "Item 3"}, Spacer[10]]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">Item 1</text>"));
    assert!(svg.contains(">Item 2</text>"));
    assert!(svg.contains(">Item 3</text>"));
  }

  #[test]
  fn row_with_spacer_list_form_renders_svg() {
    clear_state();
    let result =
      interpret_with_stdout(r#"Row[{"a", "b"}, Spacer[{20, 10}]]"#).unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">b</text>"));
  }

  #[test]
  fn row_with_string_separator_renders_svg() {
    clear_state();
    let result = interpret_with_stdout(r#"Row[{1, 2, 3}, " | "]"#).unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
    // The separator " | " should be rendered
    assert!(svg.contains(" | </text>"));
  }

  #[test]
  fn row_empty_list_passthrough() {
    clear_state();
    let result = interpret_with_stdout("Row[{}]").unwrap();
    assert_eq!(result.result, "");
  }

  #[test]
  fn row_non_list_passthrough() {
    clear_state();
    let result = interpret_with_stdout("Row[x]").unwrap();
    assert_eq!(result.result, "Row[x]");
  }
}
