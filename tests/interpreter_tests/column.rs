use super::*;

mod column_text_mode {
  use super::*;

  #[test]
  fn column_basic_list() {
    clear_state();
    assert_eq!(interpret("Column[{1, 2, 3}]").unwrap(), "Column[{1, 2, 3}]");
  }

  #[test]
  fn column_symbolic() {
    clear_state();
    assert_eq!(interpret("Column[{a, b, c}]").unwrap(), "Column[{a, b, c}]");
  }

  #[test]
  fn column_no_args() {
    clear_state();
    assert_eq!(interpret("Column[]").unwrap(), "Column[]");
  }

  #[test]
  fn column_non_list_arg() {
    clear_state();
    assert_eq!(interpret("Column[1]").unwrap(), "Column[1]");
  }

  #[test]
  fn column_with_center_alignment() {
    clear_state();
    assert_eq!(
      interpret("Column[{1, 2, 3}, Center]").unwrap(),
      "Column[{1, 2, 3}, Center]"
    );
  }

  #[test]
  fn column_with_left_alignment() {
    clear_state();
    assert_eq!(
      interpret("Column[{1, 2, 3}, Left]").unwrap(),
      "Column[{1, 2, 3}, Left]"
    );
  }

  #[test]
  fn column_with_right_alignment() {
    clear_state();
    assert_eq!(
      interpret("Column[{1, 2, 3}, Right]").unwrap(),
      "Column[{1, 2, 3}, Right]"
    );
  }

  #[test]
  fn column_head() {
    clear_state();
    assert_eq!(interpret("Head[Column[{1, 2, 3}]]").unwrap(), "Column");
  }

  #[test]
  fn column_evaluates_args() {
    clear_state();
    assert_eq!(
      interpret("Column[{1 + 1, 2 + 2}]").unwrap(),
      "Column[{2, 4}]"
    );
  }

  #[test]
  fn column_nested_in_list() {
    clear_state();
    assert_eq!(
      interpret("{Column[{1, 2}], Column[{3, 4}]}").unwrap(),
      "{Column[{1, 2}], Column[{3, 4}]}"
    );
  }

  #[test]
  fn column_with_spacing() {
    clear_state();
    assert_eq!(
      interpret("Column[{1, 2, 3}, Center, 4]").unwrap(),
      "Column[{1, 2, 3}, Center, 4]"
    );
  }
}

mod column_visual_mode {
  use super::*;

  #[test]
  fn column_renders_svg() {
    clear_state();
    let result = interpret_with_stdout("Column[{1, 2, 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">3</text>"));
  }

  #[test]
  fn column_renders_strings() {
    clear_state();
    let result =
      interpret_with_stdout("Column[{\"hello\", \"world\"}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">hello</text>"));
    assert!(svg.contains(">world</text>"));
  }

  #[test]
  fn column_center_alignment_svg() {
    clear_state();
    let result = interpret_with_stdout("Column[{1, 2, 3}, Center]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"middle\""));
  }

  #[test]
  fn column_right_alignment_svg() {
    clear_state();
    let result = interpret_with_stdout("Column[{1, 2, 3}, Right]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"end\""));
  }

  #[test]
  fn column_left_alignment_svg() {
    clear_state();
    let result = interpret_with_stdout("Column[{1, 2, 3}, Left]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"start\""));
  }

  #[test]
  fn column_empty_list_passthrough() {
    clear_state();
    let result = interpret_with_stdout("Column[{}]").unwrap();
    // Empty list can't render SVG, should pass through
    assert_eq!(result.result, "Column[{}]");
  }

  #[test]
  fn column_non_list_passthrough() {
    clear_state();
    let result = interpret_with_stdout("Column[1]").unwrap();
    assert_eq!(result.result, "Column[1]");
  }

  #[test]
  fn column_spacing_increases_height() {
    clear_state();
    // Without spacing
    let no_gap = interpret_with_stdout("Column[{1, 2, 3}]").unwrap();
    let svg_no_gap = no_gap.graphics.unwrap();
    // With spacing of 10
    let with_gap =
      interpret_with_stdout("Column[{1, 2, 3}, Left, 10]").unwrap();
    let svg_with_gap = with_gap.graphics.unwrap();

    // Extract height from SVG
    let height_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
    let h1: u32 = height_re.captures(&svg_no_gap).unwrap()[1].parse().unwrap();
    let h2: u32 = height_re.captures(&svg_with_gap).unwrap()[1]
      .parse()
      .unwrap();
    assert!(
      h2 > h1,
      "SVG with spacing should be taller: {} vs {}",
      h2,
      h1
    );
  }

  #[test]
  fn column_spacing_default_zero() {
    clear_state();
    // Column with explicit 0 spacing should match no-spacing version
    let no_gap = interpret_with_stdout("Column[{1, 2}]").unwrap();
    let with_zero = interpret_with_stdout("Column[{1, 2}, Left, 0]").unwrap();
    assert_eq!(no_gap.graphics.unwrap(), with_zero.graphics.unwrap());
  }

  #[test]
  fn column_spacing_with_center_alignment() {
    clear_state();
    let result =
      interpret_with_stdout("Column[{\"a\", \"bbb\", \"ccccc\"}, Center, 4]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"middle\""));
    // Verify all items rendered
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">bbb</text>"));
    assert!(svg.contains(">ccccc</text>"));
  }
}
