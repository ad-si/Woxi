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

  // `Column[{…, image}, Center, …]` positions its own items; a raster image
  // among them used to be typeset as text and so vanished from the picture.
  #[test]
  fn column_with_alignment_draws_an_image_item() {
    clear_state();
    let result = interpret_with_stdout(
      "Column[{\"caption\", Image[RandomReal[1, {6, 8, 3}]]}, Center, \
       ItemSize -> 32]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">caption</text>"));
    assert!(svg.contains("<image"));
    // The image is nested as a plain <svg> child, with no XML prolog left in.
    assert!(!svg.contains("<?xml"));
  }

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

  #[test]
  fn grid_cell_deploy_dynamic_shows_current_value() {
    // `Deploy[Dynamic[…]]` in a Grid cell: the Deploy wrapper comes off and
    // the Dynamic still displays its current value, not its source.
    clear_state();
    for cell in [
      "Deploy[Dynamic[Column[{1 + 1, 7}]]]",
      "Deploy[Dynamic[Text[Column[{1 + 1, 7}]]]]",
    ] {
      let result =
        interpret_with_stdout(&format!("Grid[{{{{\"a\", {cell}}}}}]")).unwrap();
      let svg = result.graphics.unwrap();
      assert!(svg.contains(">2</tspan>"), "{svg}");
      assert!(!svg.contains("Dynamic"), "{svg}");
    }
  }

  #[test]
  fn grid_cell_row_dynamic_parts_show_current_values() {
    // `Dynamic[…]` parts of a `Row` in a Grid cell — a readout line such as
    // "a = w × h = 3 × 4 = 12" — display their values, also inside the
    // `HoldForm` a `With` fills, which keeps its written (unevaluated) shape.
    clear_state();
    let result = interpret_with_stdout(
      "p = {3, 0}; Grid[{{Text[Style[Row[{With[{w = Dynamic[p[[1]]]}, \
       HoldForm[2 (w + w)]], \" = \", Dynamic[p[[1]] (7 - p[[1]])]}], 12]]}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("2(3 + 3) = 12"), "{svg}");
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

  /// A Demonstration's `Manipulate` body commonly styles its whole display
  /// as `Style[Text[Column[{…}]], size]` — the `Style` sits above a `Text`
  /// wrapper, not directly above the `Column`. This used to defeat the
  /// styled-layout pass entirely (it only looked one level down for
  /// `Column`/`Row`), so neither it nor the plain Column pass recognized
  /// the expression and the whole body fell back to unevaluated text
  /// instead of rendering.
  #[test]
  fn styled_text_wrapped_column_renders() {
    clear_state();
    let result =
      interpret_with_stdout("Style[Text[Column[{\"a\", \"bb\"}]], 20]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"), "{svg}");
    assert!(svg.contains(">bb</text>"), "{svg}");
    assert_eq!(svg.matches("font-size=\"20\"").count(), 2, "{svg}");
  }

  /// As part of a scheduled QA routine, Woxi Studio was tested against a
  /// randomly sampled Wolfram Demonstration notebook (Euclidean Algorithm:
  /// GCD and LCM) whose `Manipulate` body styles a `Column` holding
  /// `If[flag, …]` items that are `Null` while their checkbox is off.
  /// `Null` is drawn as nothing, but pushing the `Style` size into every
  /// item wrapped it as `Style[Null, 24]`, which was typeset as the word
  /// "Null". Self-authored, construct-equivalent example.
  #[test]
  fn styled_column_does_not_draw_null_items() {
    clear_state();
    let result = interpret_with_stdout(
      "flag = False; Style[Text[Column[{\"a\", If[flag, \"b\"], \"\", If[flag, 2]}]], 24]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</text>"), "{svg}");
    assert!(!svg.contains("Null"), "{svg}");
  }

  /// The same lookup-through-`Text` has to work for `Row`, and survive an
  /// extra `Pane[…]` wrapper on top — the exact nesting a Manipulate body
  /// commonly uses (`Pane[Style[Text[Row[{…}]], size], {w, h}]`).
  #[test]
  fn styled_text_wrapped_row_inside_a_pane_renders() {
    clear_state();
    let result = interpret_with_stdout(
      "Pane[Style[Text[Row[{\"a\", \"b\"}]], 20], {300, 100}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">a</tspan>"), "{svg}");
    assert!(svg.contains(">b</tspan>"), "{svg}");
  }

  /// `Pane[content, ImageSize -> {w, h}]` is the same fixed-size box as
  /// `Pane[content, {w, h}]`.
  #[test]
  fn pane_image_size_option_sets_the_box_size() {
    clear_state();
    for pane in [
      "Pane[Text[Row[{\"a\", \"b\"}]], ImageSize -> {300, 100}, \
       Alignment -> Center]",
      "Pane[Text[Row[{\"a\", \"b\"}]], {300, 100}, Alignment -> Center]",
    ] {
      let code = format!("ExportString[{pane}, \"SVG\"]");
      let svg = interpret(&code).unwrap();
      assert!(svg.contains("width=\"300\" height=\"100\""), "{svg}");
    }
  }

  /// A fixed-size `Pane` re-flows prose built from a `Row` that mixes strings,
  /// numbers and `Style`d words — even through a nested size-less `Pane` and
  /// `Text` — breaking at spaces and at explicit newlines instead of running
  /// off the edge of the box.
  #[test]
  fn sized_pane_wraps_mixed_row_prose_at_word_boundaries() {
    clear_state();
    let result = interpret_with_stdout(
      "Pane[Text[Pane[Row[{\"alpha beta gamma \", 12, \" delta \", \
       Style[\"epsilon\", Italic], \" zeta eta theta\\nsecond paragraph\"}]]], \
       {120, 200}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    // 120 / (14 * 0.6) = 14 characters per line.
    assert!(!svg.contains("alpha beta gamma"), "{svg}");
    assert!(svg.contains(">alpha beta<"), "{svg}");
    assert!(svg.contains(">second<"), "{svg}");
    assert!(svg.contains("font-style=\"italic\""), "{svg}");
  }

  /// `Style[Column[{…, Pane[text, w]}, opts], "Label", 12]` pushes the style
  /// onto each item; a `Pane` item used to come out as its own source text.
  #[test]
  fn styled_column_with_a_pane_item_draws_the_pane_content() {
    clear_state();
    let result = interpret_with_stdout(
      "Style[Column[{Graphics[Circle[]], Pane[\"caption\", 350]}, \
       Alignment -> Top, ItemSize -> {Automatic, {10, 5}}], \"Label\", 12]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("caption"), "{svg}");
    assert!(!svg.contains("Pane["), "{svg}");
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
    assert!(h2 > h1, "SVG with spacing should be taller: {h2} vs {h1}");
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
  fn column_styled_dynamic_item_shows_value() {
    clear_state();
    // `Style[Dynamic[expr], …]` is the readout idiom: the styled value, not
    // the symbolic `Dynamic[…]` text, must be drawn — alone, in a Row, or
    // inside a Column.
    let result = interpret_with_stdout(
      "Column[{Row[{Style[\"v = \", 14], Style[Dynamic[1 + 1], 14]}], \
       Style[Dynamic[Round[10/7., 0.01]], 18, Red]}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">2</tspan>"), "expected value 2, got: {svg}");
    assert!(svg.contains("1.43</"), "expected 1.43, got: {svg}");
    assert!(!svg.contains("Dynamic"), "Dynamic leaked: {svg}");
  }

  #[test]
  fn grid_cell_with_styled_dynamic_column_shows_value() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{\"title\", Column[{Row[{Style[\"v \", 14], \
       Style[Dynamic[2 + 3], 14]}], Style[Dynamic[7 * 6], 18]}, Center]}}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("5</"), "expected value 5, got: {svg}");
    assert!(svg.contains("42</"), "expected value 42, got: {svg}");
    assert!(!svg.contains("Dynamic"), "Dynamic leaked: {svg}");
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

  // ExpressionCell[expr, style] is a cell showing `expr`: a display shows
  // the expression, with no "not yet implemented" warning (six
  // demonstrations used it).
  #[test]
  fn expression_cell_is_a_known_wrapper() {
    clear_state();
    let r = interpret_with_stdout("ExpressionCell[1 + 1, \"Output\"]").unwrap();
    assert_eq!(r.result, "2");
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
  }
}

mod frameless_button_cells {
  use super::*;

  fn green_drawn(code: &str) -> bool {
    clear_state();
    let r = woxi::interpret_with_stdout(code).unwrap();
    r.graphics.is_some_and(|svg| svg.contains("rgb(0,255,0)"))
  }

  // A `Frameless` button around a picture is drawn as that picture inside a
  // layout, also when wrapped in `Dynamic`, `Pane` and `Labeled`.
  #[test]
  fn frameless_picture_button_in_grid() {
    let button = r#"Button[Graphics[{Green, Rectangle[{0, 0}]}], Null, Appearance -> "Frameless"]"#;
    assert!(green_drawn(&format!("Grid[{{{{{button}}}}}]")));
    assert!(green_drawn(&format!("Grid[{{{{Dynamic[{button}]}}}}]")));
    // The cells may be variables holding the buttons.
    assert!(green_drawn(&format!(
      "sq = Dynamic@{button}; Labeled[Dynamic[Pane[Grid[{{{{sq, sq}}}}]]], \"t\", Top]"
    )));
    assert!(green_drawn(&format!(
      "Labeled[Dynamic[Pane[Grid[{{{{{button}}}}}], ImageSize -> {{600, 414}}]], \"t\", Top]"
    )));
  }
}

mod dynamic_layout_probe {
  use super::*;

  // Whether a Dynamic layout holds a picture is probed by evaluating it;
  // the probe must not report messages of its own
  // (StockMarketReturnsByParty: a plot read data another Dynamic sets).
  #[test]
  fn probe_is_quiet() {
    clear_state();
    let r = woxi::interpret_with_stdout(
      r#"g::w = "warn"; g[] := (Message[g::w]; Grid[{{1}}]); Labeled[Dynamic[Pane[g[]]], "t", Top]"#,
    )
    .unwrap();
    assert!(
      !r.warnings.iter().any(|w| w.contains("g::w")),
      "{:?}",
      r.warnings
    );
  }
}

mod traditional_form_layout {
  use super::*;

  #[test]
  fn svg_typesets_each_item_of_a_column() {
    clear_state();
    // `TraditionalForm[Column[…]]` is a layout whose items are typeset one
    // by one; it must not print the `Column(…)` call itself.
    for code in [
      r#"ExportString[TraditionalForm[Column[{HoldForm[Integrate[Sin[x]^3, x]], HoldForm[Integrate[u^2, u]]}]], "SVG"]"#,
      r#"ExportString[TraditionalForm[Text[Style[Column[{HoldForm[Integrate[Sin[x]^3, x]], HoldForm[Integrate[u^2, u]]}, ItemSize -> 28], 18]]], "SVG"]"#,
    ] {
      let svg = interpret(code).unwrap();
      assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
      assert!(!svg.contains(">Column<"), "Column call leaked: {svg}");
      assert!(!svg.contains("TemplateBox"), "box source leaked: {svg}");
      assert!(svg.contains('∫'), "integral sign missing: {svg}");
    }
  }
}

mod row_option_in_separator_position {
  use super::*;

  /// A rule after the item list (`BaseStyle -> 20`) is an option of `Row`,
  /// not its separator.
  #[test]
  fn to_string_ignores_option_rule() {
    clear_state();
    assert_eq!(
      interpret("ToString[Row[{\"a\", 1}, BaseStyle -> 20]]").unwrap(),
      "a1"
    );
    assert_eq!(
      interpret("ToString[Row[{\"a\", 1}, \"-\", BaseStyle -> 20]]").unwrap(),
      "a-1"
    );
  }

  #[test]
  fn graphics_label_ignores_option_rule() {
    clear_state();
    let result = interpret_with_stdout(
      "ExportString[Graphics[{Inset[Row[{\"x = \", 0.5}, BaseStyle -> 20], {0, 0}]}], \"SVG\"]",
    )
    .unwrap();
    let svg = result.result;
    assert!(svg.contains("x = 0.5"), "{svg}");
    assert!(!svg.contains("BaseStyle"), "{svg}");
  }
}
