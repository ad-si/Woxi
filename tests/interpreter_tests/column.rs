use super::*;

mod column_text_mode {
  use super::*;

  #[test]
  fn column_basic_list() {
    clear_state();
    // Without a front-end (CLI / script mode, matching wolframscript) Column
    // has no built-up typeset form and prints verbatim as `Column[{…}]`.
    // The vertical-stacking layout is only produced in visual mode (see
    // `column_visual_mode` below).
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

  // Under ToString, Column stacks its elements one per line (single newline).
  // Trailing alignment/option arguments do not affect the plain-text form.
  #[test]
  fn column_to_string_stacks_lines() {
    clear_state();
    assert_eq!(interpret("ToString[Column[{1, 2, 3}]]").unwrap(), "1\n2\n3");
    assert_eq!(
      interpret("ToString[Column[{a, bb, ccc}]]").unwrap(),
      "a\nbb\nccc"
    );
    assert_eq!(interpret("ToString[Column[{1}]]").unwrap(), "1");
    assert_eq!(
      interpret("ToString[Column[{1, 2, 3}, Right]]").unwrap(),
      "1\n2\n3"
    );
  }

  #[test]
  fn column_with_center_alignment() {
    clear_state();
    // Alignment is carried along in the verbatim form in text mode.
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
  fn column_alignment_option_rule_svg() {
    // The option form `Alignment -> Center` aligns like the positional
    // `Center` argument.
    clear_state();
    let result =
      interpret_with_stdout("Column[{1, 2, 3}, Alignment -> Center]").unwrap();
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"middle\""));
  }

  #[test]
  fn column_dynamic_item_shows_current_value() {
    // A held `Dynamic[expr]` item displays as the current value of `expr`,
    // and a `Text[…]` wrapper displays as its content.
    clear_state();
    let result =
      interpret_with_stdout("Column[{Dynamic[Text[1 + 1]], 3}]").unwrap();
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">2</text>"), "{svg}");
    assert!(!svg.contains("Dynamic"), "{svg}");
  }

  /// A `Style` wrapping a layout is inherited by everything inside it, so
  /// `Style[Column[{…}], 65, Hue[…]]` — the shape of a Demonstration's
  /// `Manipulate` body — is a large, coloured column. It used to fall back
  /// to the plain-text echo of the column, at the default size and colour.
  #[test]
  fn styled_column_renders_at_the_style_size_and_colour() {
    clear_state();
    let result =
      interpret_with_stdout("Style[Column[{\"a\", \"bb\"}, Center], 65, Red]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert_eq!(svg.matches("font-size=\"65\"").count(), 2, "{svg}");
    assert_eq!(svg.matches("fill=\"rgb(255,0,0)\"").count(), 2, "{svg}");
    assert!(svg.contains("text-anchor=\"middle\""), "{svg}");
    // The canvas grows with the type: at 65 pt two rows are far taller than
    // the 44 px a default-size two-row column takes.
    let height: f64 = svg
      .split("height=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .and_then(|s| s.parse().ok())
      .unwrap();
    assert!(height > 130.0, "canvas must grow with the font size: {svg}");
  }

  /// `Invisible[expr]` keeps exactly the space `expr` occupies but paints
  /// nothing — Demonstrations hide a row with `If[show, Identity, Invisible]`
  /// and rely on the layout not jumping. It used to print as its own source.
  #[test]
  fn an_invisible_column_item_reserves_its_space_unpainted() {
    clear_state();
    let hidden =
      interpret_with_stdout("Column[{\"a\", Invisible[\"bcd\"], \"e\"}]")
        .unwrap()
        .graphics
        .unwrap();
    let shown = interpret_with_stdout("Column[{\"a\", \"bcd\", \"e\"}]")
      .unwrap()
      .graphics
      .unwrap();
    assert!(!hidden.contains("Invisible"), "{hidden}");
    assert!(hidden.contains("fill=\"none\""), "{hidden}");
    // Same geometry as the visible column — only the fill differs.
    let dims = |svg: &str| {
      svg
        .split_once('>')
        .map(|(head, _)| head.to_string())
        .unwrap()
    };
    assert_eq!(dims(&hidden), dims(&shown));
  }

  /// Either nesting order hides the item: the `Style` may sit outside the
  /// `Invisible` (which is what pushing a `Style[Column[…], …]`'s directives
  /// into the items produces) or inside it.
  #[test]
  fn an_invisible_item_stays_hidden_under_a_style() {
    clear_state();
    for src in [
      "Column[{Style[Invisible[\"bcd\"], 20, Red]}]",
      "Column[{Invisible[Style[\"bcd\", 20, Red]]}]",
    ] {
      let svg = interpret_with_stdout(src).unwrap().graphics.unwrap();
      assert!(svg.contains("fill=\"none\""), "{src}: {svg}");
      assert!(svg.contains("font-size=\"20\""), "{src}: {svg}");
      assert!(!svg.contains("Invisible"), "{src}: {svg}");
    }
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
    // An empty column has no visual form and prints verbatim (matching
    // wolframscript's `Column[{}]`).
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

  #[test]
  fn column_alignment_option_rule() {
    clear_state();
    // The option form `Alignment -> Center` must center like the
    // positional `Column[{…}, Center]` form.
    let result =
      interpret_with_stdout("Column[{1, 2, 3}, Alignment -> Center]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("text-anchor=\"middle\""));
  }

  #[test]
  fn column_dynamic_item_shows_value() {
    clear_state();
    // In visual mode `Dynamic[expr]` displays the current value of `expr`
    // (the front end re-evaluates it as dependencies change). A Column
    // item must therefore render the value, not the literal `Dynamic[…]`
    // text — the Demonstrations color-wheel notebook stacks
    // `Dynamic[Row[…]]` above its wheel graphic.
    let result = interpret_with_stdout("Column[{Dynamic[1 + 1], 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">2</text>"), "expected value 2, got: {svg}");
    assert!(svg.contains(">3</text>"));
    assert!(!svg.contains("Dynamic"));
  }

  #[test]
  fn dynamic_top_level_shows_value_in_visual_mode() {
    clear_state();
    let result = interpret_with_stdout("Dynamic[1 + 1]").unwrap();
    assert_eq!(result.result, "2");
  }

  #[test]
  fn dynamic_stays_symbolic_in_text_mode() {
    clear_state();
    // Script/CLI mode matches wolframscript: no front end, so Dynamic
    // echoes verbatim.
    assert_eq!(interpret("Dynamic[1 + 1]").unwrap(), "Dynamic[1 + 1]");
  }

  #[test]
  fn deploy_stays_symbolic_in_text_mode() {
    clear_state();
    // A front end draws Deploy's content (it only makes it
    // non-selectable), but script/CLI mode echoes the call verbatim, as
    // wolframscript does.
    assert_eq!(interpret("Deploy[1 + 1]").unwrap(), "Deploy[2]");
  }
}
