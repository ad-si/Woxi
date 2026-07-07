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
    // Script mode prints Spacer separators literally (like wolframscript);
    // only visual contexts render them as pixel gaps.
    assert_eq!(
      interpret(r#"Row[{"a", "b", "c"}, Spacer[21]]"#).unwrap(),
      "aSpacer[21]bSpacer[21]c"
    );
    assert_eq!(
      interpret(r#"Row[{"a", "b"}, Spacer[{14, 20}]]"#).unwrap(),
      "aSpacer[{14, 20}]b"
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
    // Row[{}, sep] prints as {} in wolframscript (Row[{}] prints nothing).
    assert_eq!(interpret(r#"Row[{}, Spacer[10]]"#).unwrap(), "{}");
    assert_eq!(interpret(r#"Row[{}, ","]"#).unwrap(), "{}");
    assert_eq!(interpret("Row[{}]").unwrap(), "");
  }

  #[test]
  fn evaluates_arguments() {
    clear_state();
    assert_eq!(interpret("Row[{1 + 1, 2 + 2}]").unwrap(), "24");
    assert_eq!(
      interpret("Row[{1 + 1, 2 + 2}, Spacer[7]]").unwrap(),
      "2Spacer[7]4"
    );
  }

  #[test]
  fn non_list_arg_stays_symbolic() {
    clear_state();
    assert_eq!(interpret("Row[x]").unwrap(), "Row[x]");
    assert_eq!(interpret("Row[5]").unwrap(), "Row[5]");
  }

  #[test]
  fn options_are_ignored_in_output() {
    clear_state();
    assert_eq!(interpret("Row[{1, 2}, ImageSize -> 400]").unwrap(), "12");
    assert_eq!(
      interpret(
        "Row[{1, 2}, Alignment -> {{Right, Left}, Center}, ImageSize -> 400]"
      )
      .unwrap(),
      "12"
    );
    // A rule in separator position is an option, not a separator.
    assert_eq!(interpret("Row[{1, 2}, a -> b]").unwrap(), "12");
  }

  #[test]
  fn separator_with_options() {
    clear_state();
    assert_eq!(
      interpret(r#"Row[{1, 2}, "|", ImageSize -> 400]"#).unwrap(),
      "1|2"
    );
    assert_eq!(
      interpret(r#"Row[{"a", "b"}, Spacer[7], Alignment -> Center]"#).unwrap(),
      "aSpacer[7]b"
    );
  }

  #[test]
  fn extra_non_rule_arg_stays_symbolic() {
    clear_state();
    assert_eq!(
      interpret(r#"Row[{1, 2}, "|", "x"]"#).unwrap(),
      "Row[{1, 2}, |, x]"
    );
  }

  #[test]
  fn graphics_item_shows_placeholder() {
    clear_state();
    assert_eq!(
      interpret(r#"Row[{Graphics[{Red, Disk[]}], "x"}]"#).unwrap(),
      "-Graphics-x"
    );
    assert_eq!(
      interpret(
        r#"Row[
          {
            Graphics[{Red, Disk[]}, ImageSize -> {18, 18}, PlotRange -> 2],
            Style["Hey there", Gray]
          },
          Alignment -> {{Right, Left}, Center}, ImageSize -> 400
        ]"#
      )
      .unwrap(),
      "-Graphics-Hey there"
    );
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
  fn row_embeds_graphics_items() {
    clear_state();
    let result = interpret_with_stdout(
      r#"Row[{Graphics[{Red, Disk[]}, ImageSize -> {18, 18}], "label"}]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    // The disk graphic is embedded as a nested <svg>, not shown as text.
    assert!(svg.contains("<svg x="));
    assert!(!svg.contains("Graphics["));
    assert!(svg.contains(">label</text>"));
  }

  #[test]
  fn row_applies_style_color() {
    clear_state();
    let result = interpret_with_stdout(r#"Row[{Style["hey", Gray]}]"#).unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(128,128,128)\""));
    assert!(svg.contains(">hey</text>"));
  }

  #[test]
  fn row_honors_image_size_and_alignment() {
    clear_state();
    let result = interpret_with_stdout(
      r#"Row[
        {
          Graphics[{Red, Disk[]}, ImageSize -> {18, 18}, PlotRange -> 2],
          Style["Hey there", Gray]
        },
        Alignment -> {{Right, Left}, Center}, ImageSize -> 400
      ]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    // ImageSize -> 400 widens the canvas to 400.
    assert!(svg.starts_with("<svg width=\"400\""));
    // The disk graphic is embedded, the styled text is gray.
    assert!(svg.contains("<svg x="));
    assert!(svg.contains("fill=\"rgb(128,128,128)\""));
    assert!(svg.contains(">Hey there</text>"));
  }

  #[test]
  fn row_alignment_right_shifts_content() {
    clear_state();
    let result = interpret_with_stdout(
      r#"Row[{"ab"}, Alignment -> Right, ImageSize -> 200]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.starts_with("<svg width=\"200\""));
    // Content block is right-aligned: the text center sits near the
    // right edge (200 - width/2), far past the midpoint.
    let x_attr = svg
      .split("<text x=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .and_then(|s| s.parse::<f64>().ok())
      .unwrap();
    assert!(x_attr > 150.0, "text x = {x_attr}");
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
