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

  /// The front end reads a string alignment (`Column[…, "Center"]`) like
  /// the symbol: every item is centered, not left-aligned.
  #[test]
  fn column_string_alignment_matches_symbol() {
    for (string, symbol) in [("\"Center\"", "Center"), ("\"Right\"", "Right")] {
      clear_state();
      let code = |align: &str| {
        format!(
          "Column[{{\"a long line\", \"x\", Graphics[Circle[]]}}, {align}]"
        )
      };
      let by_string = interpret_with_stdout(&code(string)).unwrap().graphics;
      clear_state();
      let by_symbol = interpret_with_stdout(&code(symbol)).unwrap().graphics;
      assert!(by_string.is_some());
      assert_eq!(by_string, by_symbol, "{string}");
    }
  }

  /// Count the horizontal rules (`<line … y1 == y2 …>`) in an SVG.
  fn horizontal_lines(svg: &str) -> Vec<f64> {
    svg
      .split("<line ")
      .skip(1)
      .filter_map(|l| {
        let attr = |k: &str| -> Option<f64> {
          let start = l.find(&format!(" {k}=\""))? + k.len() + 3;
          l[start..].split('"').next()?.parse().ok()
        };
        let (y1, y2) = (attr("y1")?, attr("y2")?);
        (y1 == y2).then_some(y1)
      })
      .collect()
  }

  // `Dividers` on a `Column` rules lines between its items, the way it
  // does on a one-column `Grid`; it used to be ignored.
  #[test]
  fn column_dividers_per_position_rule_lines() {
    clear_state();
    let result = interpret_with_stdout(
      "Column[{\"a\", \"b\", \"c\", \"d\", \"e\"}, Left, \
       Dividers -> {None, {False, True, False, True}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    let lines = horizontal_lines(&svg);
    assert_eq!(lines.len(), 2, "{svg}");
    for item in ["a", "b", "c", "d", "e"] {
      assert!(svg.contains(&format!(">{item}</text>")), "{svg}");
    }
    // Left alignment is kept.
    assert!(svg.contains("text-anchor=\"start\""), "{svg}");
  }

  // A fixed-size `Pane` item keeps its box in a ruled column (a `Grid`
  // layout); it used to be peeled, so a wide picture drew at its own size.
  #[test]
  fn ruled_column_keeps_a_fixed_size_pane_box() {
    clear_state();
    let result = interpret_with_stdout(
      "Column[{Pane[Graphics[Disk[], ImageSize -> 720], \
       ImageSize -> {500, 35}], \"x\"}, Dividers -> All]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"500.0\" height=\"35.0\""), "{svg}");
    assert!(!svg.contains("width=\"720"), "{svg}");
  }

  #[test]
  fn grid_cell_keeps_a_fixed_size_pane_box() {
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{Pane[\"abc\", {80, 30}], 1}, {2, 3}}, Frame -> All]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("width=\"80.0\" height=\"30.0\""), "{svg}");
  }

  /// The `fill` colors of the `<rect>`s in `svg`, top to bottom.
  fn rect_fills(svg: &str) -> Vec<String> {
    svg
      .split("<rect")
      .skip(1)
      .filter_map(|r| {
        let start = r.find(" fill=\"")? + 7;
        Some(r[start..].split('"').next()?.to_string())
      })
      .collect()
  }

  // As part of a scheduled QA routine, Woxi Studio was tested against a
  // randomly sampled Wolfram Demonstration notebook whose word list is a
  // `Column` striped with a cyclic `Background -> {{c1, c2}}`; the row
  // fills were dropped (only a single color filled the column).
  #[test]
  fn column_cyclic_background_stripes_rows() {
    clear_state();
    let result = interpret_with_stdout(
      "Column[{\"a\", \"b\", \"c\", \"d\", \"e\"}, Left, \
       Background -> {{Yellow, Cyan}}, Spacings -> 1]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    let (y, c) = ("rgb(255,255,0)", "rgb(0,255,255)");
    assert_eq!(rect_fills(&svg), [y, c, y, c, y], "{svg}");
    for item in ["a", "b", "c", "d", "e"] {
      assert!(svg.contains(&format!(">{item}</text>")), "{svg}");
    }
    assert!(svg.contains("text-anchor=\"start\""), "{svg}");
  }

  // A flat background list colors successive rows once each.
  #[test]
  fn column_background_list_colors_successive_rows() {
    clear_state();
    let result =
      interpret_with_stdout("Column[{1, 2, 3}, Background -> {Yellow, Cyan}]")
        .unwrap();
    let svg = result.graphics.unwrap();
    assert_eq!(
      rect_fills(&svg),
      ["rgb(255,255,0)", "rgb(0,255,255)"],
      "{svg}"
    );
  }

  #[test]
  fn column_dividers_all_rules_every_line() {
    clear_state();
    let result =
      interpret_with_stdout("Column[{1, 2, 3}, Dividers -> All]").unwrap();
    let svg = result.graphics.unwrap();
    // Above, between and below the three items.
    assert_eq!(horizontal_lines(&svg).len(), 4, "{svg}");
  }

  #[test]
  fn column_dividers_center_rules_inner_lines() {
    clear_state();
    let result =
      interpret_with_stdout("Column[{1, 2, 3}, Dividers -> Center]").unwrap();
    let svg = result.graphics.unwrap();
    assert_eq!(horizontal_lines(&svg).len(), 2, "{svg}");
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
      // (`Text[…]` sets the column in the text face, as `<text>` lines.)
      assert!(
        svg.contains(">2</tspan>") || svg.contains(">2</text>"),
        "{svg}"
      );
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

  #[test]
  fn grid_cell_text_styled_column_uses_text_face() {
    // `Text[Style[Column[…], size]]` in a Grid cell is set in the
    // proportional text face, as `Text[Column[…]]` and
    // `Text[Style[Row[…], size]]` are, also when a line mixes strings and
    // numbers; it used to fall back to the monospace face of typeset output.
    clear_state();
    let result = interpret_with_stdout(
      "Grid[{{Text[Style[Column[{\"\", Row[{\"weight is \", 7, \".\"}]}], \
       18]], SpanFromLeft}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.contains("weight is"), "{svg}");
    assert!(svg.contains("font-size=\"18\""), "{svg}");
    assert!(!svg.contains("monospace"), "{svg}");
    assert!(svg.contains("sans-serif"), "{svg}");
  }

  /// A `Style` wrapping a layout is inherited by everything inside it, so
  /// `Style[Column[{…}], 65, Hue[…]]` — the shape of a Demonstration's
  /// `Manipulate` body — is a large, colored column. It used to fall back
  /// to the plain-text echo of the column, at the default size and color.
  #[test]
  fn styled_column_renders_at_the_style_size_and_color() {
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

  /// Regression: in a `Text[Column[…]]`, an item styled as
  /// `Style[TraditionalForm[Grid[…, Frame -> All]], 10]` took the text face
  /// as a second `Style` around its own, which hid the grid from the layout
  /// renderer, so it was written out as the one-line `Grid[{{…}}]` source.
  #[test]
  fn text_column_lays_out_a_styled_traditional_form_grid() {
    clear_state();
    for item in [
      "Style[TraditionalForm[Grid[{{1, x + 1}, {2, 3}}, Frame -> All]], 10]",
      "Style[StandardForm[Grid[{{1, x + 1}, {2, 3}}, Frame -> All]], 10]",
    ] {
      let result = interpret_with_stdout(&format!(
        "Text[Column[{{\"head\", {item}}}, Alignment -> {{Center}}]]"
      ))
      .unwrap();
      assert_eq!(result.result, "-Graphics-");
      let svg = result.graphics.unwrap();
      assert!(!svg.contains("Grid["), "{svg}");
      // The 2x2 grid's frame: three horizontal and three vertical rules.
      assert_eq!(svg.matches("<line").count(), 6, "{svg}");
      // The grid's cells take the style's size; only the heading keeps
      // the default one.
      assert!(svg.contains("font-size=\"10"), "{svg}");
      assert_eq!(svg.matches("font-size=\"14").count(), 1, "{svg}");
    }
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
    assert!(svg.contains(">a</text>"), "{svg}");
    assert!(svg.contains(">b</text>"), "{svg}");
    assert!(svg.contains("font-size=\"20\""), "{svg}");
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

  /// A fixed-size `Pane` re-flows a `Row[list, sep]` (a word list joined by
  /// ", ") into lines that break after the separators, each set as one
  /// string, instead of one line far wider than the box — which under
  /// `ImageSizeAction -> "ShrinkToFit"` had shrunk the text to a sliver.
  #[test]
  fn sized_pane_wraps_a_row_with_a_separator() {
    clear_state();
    let svg = interpret(
      "ExportString[Pane[Style[Row[Table[\"WORD\" <> ToString[i], {i, 30}], \
       \", \"], 18], {500, 300}, Alignment -> {Left, Center}, \
       ImageSizeAction -> \"ShrinkToFit\"], \"SVG\"]",
    )
    .unwrap();
    assert!(
      svg.starts_with("<svg width=\"500\" height=\"300\""),
      "{svg}"
    );
    // 500 / (18 * 0.632) = 43 characters per line.
    assert!(
      svg.contains(">WORD1, WORD2, WORD3, WORD4, WORD5, WORD6,<"),
      "{svg}"
    );
    assert!(svg.contains(">WORD7, WORD8, WORD9, "), "{svg}");
    assert!(!svg.contains("font-size=\"14\""), "{svg}");
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

  fn column_svg_height(code: &str) -> f64 {
    clear_state();
    let svg = interpret_with_stdout(code).unwrap().graphics.unwrap();
    let height_re = regex::Regex::new(r#"height="([\d.]+)""#).unwrap();
    height_re.captures(&svg).unwrap()[1].parse().unwrap()
  }

  #[test]
  fn column_spacing_zero_is_tighter_than_default() {
    // The default gap between a column's lines is a small fixed offset,
    // so an explicit spacing of 0 sets the lines closer together.
    let default = column_svg_height("Column[{1, 2}]");
    let zero = column_svg_height("Column[{1, 2}, Left, 0]");
    assert!(zero < default, "{zero} vs {default}");
  }

  #[test]
  fn column_spacing_is_measured_between_lines_of_text() {
    // `s` ems separate the lines themselves (each 0.93 em tall), not
    // their padded boxes: at 14 pt, lines 3.93 em = 55 pt apart.
    let pitch = |code: &str| {
      let svg = interpret_with_stdout(code).unwrap().graphics.unwrap();
      let y_re = regex::Regex::new(r#"<text x="[^"]*" y="([\d.]+)""#).unwrap();
      let ys: Vec<f64> = y_re
        .captures_iter(&svg)
        .map(|c| c[1].parse().unwrap())
        .collect();
      assert_eq!(ys.len(), 3, "{svg}");
      (ys[1] - ys[0], ys[2] - ys[1])
    };
    clear_state();
    assert_eq!(
      pitch("Column[{\"a\", \"b\", \"c\"}, Left, 3]"),
      (55.0, 55.0)
    );
    assert_eq!(
      pitch("Column[{\"a\", \"b\", \"c\"}, Spacings -> 3]"),
      (55.0, 55.0)
    );
    assert_eq!(
      pitch("Column[{\"a\", \"b\", \"c\"}, Left, 1]"),
      (27.0, 27.0)
    );
  }

  #[test]
  fn column_spacing_of_rows_matches_spacing_of_strings() {
    // A `Row` of text items is a line of text too, even though the column
    // draws it as a picture first.
    let strings =
      column_svg_height("Column[{\"a = 1\", \"b = 2\", \"c = 3\"}, Left, 3]");
    let rows = column_svg_height(
      "Column[{Row[{\"a\", \" = \", 1}], Row[{\"b\", \" = \", 2}], \
       Row[{\"c\", \" = \", 3}]}, Left, 3]",
    );
    assert_eq!(rows, strings);
    assert_eq!(rows, 133.0);
  }

  #[test]
  fn column_spacing_between_pictures_is_the_full_gap() {
    // A picture fills its box, so all of `s` ems stays between two of them.
    let tight = column_svg_height(
      "Column[{Graphics[Disk[], ImageSize -> 50], Graphics[Disk[], ImageSize -> 50]}, Left, 0]",
    );
    let spaced = column_svg_height(
      "Column[{Graphics[Disk[], ImageSize -> 50], Graphics[Disk[], ImageSize -> 50]}, Left, 2]",
    );
    assert_eq!(spaced - tight, 28.0);
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
  fn column_alignment_horizontal_vertical_pair() {
    clear_state();
    // `Alignment -> {h, v}` gives the items' horizontal alignment first:
    // `{Center, Top}` centers them like a bare `Alignment -> Center`, and
    // `{Right, Top}` right-aligns them.
    for (spec, anchor) in [("{Center, Top}", "middle"), ("{Right, Top}", "end")]
    {
      let result = interpret_with_stdout(&format!(
        "Column[{{1, 22222, 3}}, Alignment -> {spec}]"
      ))
      .unwrap();
      assert_eq!(result.result, "-Graphics-");
      let svg = result.graphics.unwrap();
      assert!(
        svg.contains(&format!("text-anchor=\"{anchor}\"")),
        "{spec}: {svg}"
      );
    }
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
    assert!(svg.contains(">2</text>"), "expected value 2, got: {svg}");
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

  /// The left edge a line of a left-aligned column starts at: the `x` of
  /// a text item's `<text>`, or of a laid-out item's nested `<svg>`.
  fn line_starts(svg: &str) -> Vec<f64> {
    let body = svg.split_once('\n').map_or(svg, |(_, rest)| rest);
    body
      .lines()
      .filter(|l| {
        (l.starts_with("<text ") && l.contains("text-anchor=\"start\""))
          || l.starts_with("<svg x=")
      })
      .filter_map(|l| {
        let rest = l.split_once(" x=\"")?.1;
        rest.split('"').next()?.parse().ok()
      })
      .collect()
  }

  /// A line of math (a `Row`) starts where a line of prose does, so the
  /// lines of a left-aligned column share one edge.
  #[test]
  fn text_and_row_items_share_the_left_edge() {
    for code in [
      r#"Column[{"Since", Row[{x, " = ", y}]}]"#,
      r#"Column[{Row[{a + b, " = ", c}], "follows", Row[{x}]}]"#,
    ] {
      clear_state();
      let svg = interpret(&format!(r#"ExportString[{code}, "SVG"]"#)).unwrap();
      let starts = line_starts(&svg);
      assert!(starts.len() >= 2, "{code}: {svg}");
      assert!(starts.iter().all(|&x| x == starts[0]), "{code}: {starts:?}");
      assert!(starts[0] > 0.0, "{code}: {starts:?}");
    }
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
      // The integral sign is drawn as a path.
      assert!(
        svg.contains("stroke-linecap=\"round\""),
        "integral sign missing: {svg}"
      );
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

mod column_item_size_wrapping {
  use super::*;

  /// A fixed `ItemSize` width breaks prose wider than the column into
  /// lines at word boundaries, for a string item and for a `Row` of text.
  #[test]
  fn wide_text_wraps_to_item_width() {
    for item in [
      "\"b = 0, so the square is already complete.\"",
      "Row[{\"b\", \" = \", 0, \", so the square is already complete.\"}]",
    ] {
      clear_state();
      let out = interpret(&format!(
        "ExportString[TraditionalForm[Column[{{{item}}}, \
         Alignment -> {{Left, Top}}, ItemSize -> {{{{18}}, Automatic}}]], \
         \"SVG\"]"
      ))
      .unwrap();
      assert!(out.contains(">b = 0, so the square is</text>"), "{out}");
      assert!(out.contains(">already complete.</text>"), "{out}");
    }
  }

  /// Text that fits keeps its single line.
  #[test]
  fn narrow_text_stays_on_one_line() {
    clear_state();
    let out = interpret(
      "ExportString[Column[{Row[{\"b\", \" = \", 0}]}, ItemSize -> 18], \"SVG\"]",
    )
    .unwrap();
    assert!(out.contains(">b</text>"), "{out}");
    assert!(out.contains("height=\"22\""), "{out}");
  }

  /// A `Text[…]` item is measured in the proportional text face: a line
  /// that fits the column in that face stays whole, although it holds more
  /// characters than the column has monospace columns, and it stays sans.
  #[test]
  fn text_item_fitting_in_proportional_face_is_not_wrapped() {
    clear_state();
    let out = interpret(
      "ExportString[Column[{Text[\"the quick brown fox jumps over the lazy \
       dog again.\"]}, ItemSize -> {30, Automatic}], \"SVG\"]",
    )
    .unwrap();
    assert!(
      out
        .contains(">the quick brown fox jumps over the lazy dog again.</text>"),
      "{out}"
    );
    assert!(
      !out.contains(
        "<text x=\"6.0\" y=\"11.0\" font-family=\"Atkinson Hyperlegible Mono"
      ),
      "{out}"
    );
  }

  /// A `Text[…]` item too wide for the column wraps at word boundaries by
  /// its proportional width, keeps its own line breaks, and every line is
  /// drawn in the text face, flush with the other items.
  #[test]
  fn wide_text_item_wraps_in_proportional_face() {
    clear_state();
    let out = interpret(
      "ExportString[Column[{Text[\"first line\\nthe quick brown fox jumps \
       over the lazy dog again.\"], Text[\"x\"]}, \
       ItemSize -> {20, Automatic}], \"SVG\"]",
    )
    .unwrap();
    let lines: Vec<&str> =
      out.lines().filter(|l| l.starts_with("<text")).collect();
    assert_eq!(lines.len(), 4, "{out}");
    assert!(lines[0].contains(">first line</text>"), "{out}");
    assert!(
      lines[1].contains(">the quick brown fox jumps over the lazy dog</text>"),
      "{out}"
    );
    assert!(lines[2].contains(">again.</text>"), "{out}");
    for line in lines {
      assert!(line.starts_with("<text x=\"0.0\""), "{out}");
      assert!(
        line.contains("font-family=\"Atkinson Hyperlegible Next, sans-serif\""),
        "{out}"
      );
    }
  }
}

mod stretched_button_items {
  use super::*;

  /// The width of the button plate (the first rounded `rect`) in `svg`.
  fn plate_width(svg: &str) -> f64 {
    let at = svg.find("rx=\"4\"").expect("button plate");
    let rect = &svg[svg[..at].rfind("<rect").unwrap()..at];
    let w = rect.split("width=\"").nth(1).unwrap();
    w[..w.find('"').unwrap()].parse().unwrap()
  }

  // A text-labeled `Button` stacked in a `Column` with a wider item
  // stretches to the column's width, as the FrontEnd lays it out.
  #[test]
  fn button_above_a_picture_spans_the_column() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Column[{Button["go", x = 1], Graphics[Circle[], ImageSize -> 240]}], "SVG"]"#,
    )
    .unwrap();
    assert!(!svg.contains(">Button<"), "button printed as source");
    assert!(plate_width(&svg) > 230.0, "{}", plate_width(&svg));
  }

  #[test]
  fn button_above_long_text_spans_the_column() {
    clear_state();
    let r = woxi::interpret_with_stdout(
      r#"Column[{Button["go", x = 1], "a considerably longer line of text"}]"#,
    )
    .unwrap();
    let svg = r.graphics.expect("column graphic");
    let (narrow, _) = woxi::functions::graphics::parse_svg_wh(&svg);
    assert!(plate_width(&svg) > narrow - 2.0, "{svg}");
  }

  // An explicit `ImageSize` keeps the button at its requested width.
  #[test]
  fn sized_button_keeps_its_width() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Column[{Button["go", x = 1, ImageSize -> 80], Graphics[Circle[], ImageSize -> 240]}], "SVG"]"#,
    )
    .unwrap();
    assert!(
      (plate_width(&svg) - 79.0).abs() < 1.0,
      "{}",
      plate_width(&svg)
    );
  }

  // A bare `Button` exports as its plate, not its source text.
  #[test]
  fn bare_button_exports_as_plate() {
    clear_state();
    let svg = interpret(r#"ExportString[Button["go", x = 1], "SVG"]"#).unwrap();
    assert!(!svg.contains(">Button<"), "{svg}");
    assert!(plate_width(&svg) < 60.0, "{}", plate_width(&svg));
  }
}

mod column_text_cell_in_grid {
  use super::*;

  fn svg_width(svg: &str) -> f64 {
    let start = svg.find("width=\"").unwrap() + 7;
    let end = start + svg[start..].find('"').unwrap();
    svg[start..end].parse().unwrap()
  }

  fn grid_of_column(items: &str) -> String {
    clear_state();
    interpret(&format!(
      "ExportString[Grid[{{{{Text[Column[{{{items}}}]]}}}}], \"SVG\"]"
    ))
    .unwrap()
  }

  /// A column of text lines in a grid cell is as wide as its widest line,
  /// not as all of its lines laid end to end — a heading ending in a line
  /// break must not stretch it either.
  #[test]
  fn width_is_the_widest_line() {
    let one =
      grid_of_column(r#"Style["Head\n", Bold], "aaaaaaaaaaaaaaaaaaaa""#);
    let many = grid_of_column(
      r#"Style["Head\n", Bold], "aaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbb", "cccccccccccccccccccc""#,
    );
    assert_eq!(svg_width(&one), svg_width(&many), "{many}");
    assert!(svg_width(&many) < 250.0, "{many}");
  }

  /// The lines of a `Column` are left-aligned (its default alignment).
  #[test]
  fn lines_are_left_aligned() {
    let svg = grid_of_column(r#""a", "bbbbbbbbbbbb", "cc""#);
    assert!(svg.contains("text-anchor=\"start\""), "{svg}");
    assert!(!svg.contains("text-anchor=\"middle\""), "{svg}");
  }

  /// An empty second argument — the `Null` a trailing comma leaves in
  /// `Column[{…}, ]` — sets the lines flush right, like `Right`.
  #[test]
  fn empty_alignment_argument_is_right_aligned() {
    let svg = |rest: &str| {
      interpret(&format!(
        "ExportString[Column[{{\"a\", \"bbbbbbbbbbbb\"}}{rest}], \"SVG\"]"
      ))
      .unwrap()
    };
    assert_eq!(svg(", "), svg(", Right"));
    assert_eq!(svg(", Null"), svg(", Right"));
    assert_ne!(svg(", "), svg(""));
  }

  /// A `FontFamily` on one line of the column reaches that line's text.
  #[test]
  fn item_font_family_is_kept() {
    let svg = grid_of_column(
      r#""Title", Style["0123456789", FontFamily -> "Courier", 10]"#,
    );
    // The item keeps its own face and size, whether the column is set as
    // `<tspan>` runs or (in the text face of `Text[…]`) as `<text>` lines.
    let line = svg
      .split('<')
      .find(|t| t.ends_with(">0123456789"))
      .unwrap_or_else(|| panic!("no item text in {svg}"));
    assert!(
      line.contains("font-family=\"Courier, monospace\"")
        && line.contains("font-size=\"10\""),
      "{svg}"
    );
  }
}

mod row_part_font_family {
  use super::*;

  fn svg_width(svg: &str) -> f64 {
    let start = svg.find("width=\"").unwrap() + 7;
    let end = start + svg[start..].find('"').unwrap();
    svg[start..end].parse().unwrap()
  }

  /// A `Row` cell whose parts break onto several lines is as wide as its
  /// widest line.
  #[test]
  fn row_cell_with_line_breaks_is_widest_line_wide() {
    clear_state();
    let broken = interpret(
      r#"ExportString[Grid[{{Row[{"aaaaaaaaaa\n", "bbbbbbbbbb\ncc"}]}}], "SVG"]"#,
    )
    .unwrap();
    let single =
      interpret(r#"ExportString[Grid[{{"aaaaaaaaaa"}}], "SVG"]"#).unwrap();
    assert_eq!(svg_width(&broken), svg_width(&single), "{broken}");
  }

  /// `Style[…, FontFamily -> f]` on a part of a `Row` cell sets that part
  /// in `f`.
  #[test]
  fn styled_font_family_in_row_cell() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Grid[{{Row[{"x ", Style["y", FontFamily -> "Times"]}]}}], "SVG"]"#,
    )
    .unwrap();
    assert!(
      svg.contains("<tspan font-family=\"Times, serif\">y</tspan>"),
      "{svg}"
    );
  }
}

mod text_display_face {
  use super::*;

  fn svg_width(svg: &str) -> f64 {
    let start = svg.find("width=\"").unwrap() + 7;
    let end = start + svg[start..].find('"').unwrap();
    svg[start..end].parse().unwrap()
  }

  /// A `Text[…]` title in a fixed-size `Pane` is set in the proportional
  /// text face: a line that would overflow at the monospace advance still
  /// fits on one line, and it keeps that face.
  #[test]
  fn sized_pane_keeps_text_title_on_one_proportional_line() {
    clear_state();
    let out = interpret(
      "ExportString[Pane[Column[{Text[Style[\"   Quarterly sales overview, \
       revenue by region and product\", 16]], Spacer[10], 1}, \
       Alignment -> Center], {525, 425}, Alignment -> {Center, Top}], \"SVG\"]",
    )
    .unwrap();
    assert!(
      out.contains(
        "font-family=\"Atkinson Hyperlegible Next, sans-serif\" \
         font-size=\"16\""
      ),
      "{out}"
    );
    assert!(
      out.contains(
        ">   Quarterly sales overview, revenue by region and product<"
      ),
      "{out}"
    );
  }

  /// When a `Text[…]` line in a fixed-size `Pane` does need breaking, the
  /// lines are measured and drawn in the proportional text face.
  #[test]
  fn sized_pane_wraps_text_in_the_proportional_face() {
    clear_state();
    let out = interpret(
      "ExportString[Pane[Text[\"aaa bbb ccc ddd eee fff ggg hhh iii jjj\"], \
       {150, 100}], \"SVG\"]",
    )
    .unwrap();
    assert!(out.contains(">aaa bbb ccc ddd eee fff</text>"), "{out}");
    assert!(out.contains(">ggg hhh iii jjj</text>"), "{out}");
    assert!(!out.contains("Hyperlegible Mono"), "{out}");
  }

  /// A `Column` sizes a plain `Text` string glyph by glyph: narrow letters
  /// take less room than wide ones.
  #[test]
  fn column_measures_text_items_by_glyph() {
    clear_state();
    let narrow = interpret(
      "ExportString[Column[{Text[\"iiiiiiiiiiiiiiiiiiii\"]}], \"SVG\"]",
    )
    .unwrap();
    let wide = interpret(
      "ExportString[Column[{Text[\"MMMMMMMMMMMMMMMMMMMM\"]}], \"SVG\"]",
    )
    .unwrap();
    assert!(
      svg_width(&narrow) * 2.0 < svg_width(&wide),
      "{narrow}\n{wide}"
    );
  }

  /// `Text@Grid[…]` sets its text cells in the proportional text face, so
  /// its columns are sized by the glyphs' own widths — narrower than the
  /// monospace columns of a bare `Grid` — also inside a `Row`.
  #[test]
  fn text_grid_columns_use_proportional_widths() {
    let grid = "Grid[{{\"fit statistic  \", \"value\"}, \
                {\"alpha\", NumberForm[0.00051234, {10, 4}]}}, \
                Alignment -> Left, Frame -> True]";
    clear_state();
    let plain = interpret(&format!("ExportString[{grid}, \"SVG\"]")).unwrap();
    let text =
      interpret(&format!("ExportString[Text[{grid}], \"SVG\"]")).unwrap();
    let in_row = interpret(&format!(
      "ExportString[Row[{{Spacer[20], Text[{grid}]}}], \"SVG\"]"
    ))
    .unwrap();
    let (plain_w, text_w) = (svg_width(&plain), svg_width(&text));
    assert!(text_w < plain_w * 0.8, "{plain_w} vs {text_w}");
    assert_eq!(svg_width(&in_row), text_w + 20.0, "{in_row}");
    // The value column starts right after the widest label (its drawn extent).
    assert!(text.contains("<text x=\"90.3\""), "{text}");
  }

  /// A `Text@Grid` cell that is a `Row` of plain items (a bold number
  /// between strings) or a bold `Style` string is measured in the text face
  /// too, glyph by glyph, as its plain-string neighbors are — not at the
  /// monospace advance, which spread such a grid far too wide.
  #[test]
  fn text_grid_row_and_style_cells_use_proportional_widths() {
    clear_state();
    let width_of = |code: &str| {
      svg_width(&interpret(&format!("ExportString[{code}, \"SVG\"]")).unwrap())
    };
    for cell in [
      "Row[{\"  \", Style[1, Bold], \": 12-34-65\"}]",
      "Row[{\"ab\", \"cd\"}, \"-\"]",
      "Style[\"iiiiiiii\", Bold]",
    ] {
      let plain = width_of(&format!("Grid[{{{{{cell}}}}}]"));
      let text = width_of(&format!("Text@Grid[{{{{{cell}}}}}]"));
      let in_row = width_of(&format!("Text@Row[{{Grid[{{{{{cell}}}}}]}}]"));
      assert!(text < plain, "{cell}: {plain} vs {text}");
      assert_eq!(in_row, text, "{cell}");
    }
    // A row cell sets as wide as the plain string of its joined text (plus
    // a little for the bold digit).
    let row =
      width_of("Text@Grid[{{Row[{\"  \", Style[1, Bold], \": 12\"}]}}]");
    let joined = width_of("Text@Grid[{{\"  1: 12\"}}]");
    assert!(row >= joined && row <= joined + 2.0, "{row} vs {joined}");
    // Narrow glyphs make a narrower row cell than wide ones.
    let narrow = width_of("Text@Grid[{{Row[{\"iiii\", \"iiii\"}]}}]");
    let wide = width_of("Text@Grid[{{Row[{\"MMMM\", \"MMMM\"}]}}]");
    assert!(narrow * 2.0 < wide, "{narrow} vs {wide}");
  }
}

mod column_item_style_and_size {
  use super::*;

  fn svg_dims(svg: &str) -> (f64, f64) {
    let attr = |name: &str| -> f64 {
      let start = svg.find(&format!("{name}=\"")).unwrap() + name.len() + 2;
      let end = start + svg[start..].find('"').unwrap();
      svg[start..end].parse().unwrap()
    };
    (attr("width"), attr("height"))
  }

  /// `ItemStyle -> Directive[…]` styles every item of a `Column`: size,
  /// weight and color reach each item's text.
  #[test]
  fn single_item_style_applies_to_every_item() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Column[{"a", "b"}, ItemStyle -> Directive[20, Bold, Blue]], "SVG"]"#,
    )
    .unwrap();
    assert_eq!(svg.matches("font-size=\"20\"").count(), 2, "{svg}");
    assert_eq!(svg.matches("font-weight=\"bold\"").count(), 2, "{svg}");
    assert_eq!(svg.matches("fill=\"rgb(0,0,255)\"").count(), 2, "{svg}");
  }

  /// A list of styles styles the items in turn; a sublist repeats.
  #[test]
  fn list_item_style_styles_items_in_turn() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Column[{"a", "b", "c"}, ItemStyle -> {Red, {Blue}}], "SVG"]"#,
    )
    .unwrap();
    assert_eq!(svg.matches("fill=\"rgb(255,0,0)\"").count(), 1, "{svg}");
    assert_eq!(svg.matches("fill=\"rgb(0,0,255)\"").count(), 2, "{svg}");
  }

  /// The option itself never shows up as an item or as text.
  #[test]
  fn item_style_with_alignment_and_spacing() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Column[{"x", "y"}, Center, 0, ItemStyle -> Directive[18, Red]], "SVG"]"#,
    )
    .unwrap();
    assert!(!svg.contains("ItemStyle"), "{svg}");
    assert_eq!(svg.matches("font-size=\"18\"").count(), 2, "{svg}");
  }

  /// A fixed `ItemSize` width is the whole width of the column.
  #[test]
  fn fixed_item_width_is_the_column_width() {
    for w in [3, 5] {
      clear_state();
      let svg = interpret(&format!(
        r#"ExportString[Column[{{"a", "b"}}, ItemSize -> {w}], "SVG"]"#
      ))
      .unwrap();
      assert_eq!(svg_dims(&svg).0, (w * 14) as f64, "{svg}");
    }
  }

  /// Stacked framed boxes are set apart like stacked pictures, whether or
  /// not the `Column` has an alignment argument.
  #[test]
  fn stacked_boxes_have_the_default_gap() {
    clear_state();
    let one = interpret(r#"ExportString[Framed["a"], "SVG"]"#).unwrap();
    let (_, h) = svg_dims(&one);
    for align in ["", ", Left", ", Center"] {
      clear_state();
      let svg = interpret(&format!(
        r#"ExportString[Column[{{Framed["a"], Framed["b"]}}{align}], "SVG"]"#
      ))
      .unwrap();
      assert_eq!(svg_dims(&svg).1, 2.0 * h + 6.0, "{align}: {svg}");
    }
  }

  /// An explicit spacing of 0 keeps the boxes touching.
  #[test]
  fn explicit_zero_spacing_keeps_boxes_touching() {
    clear_state();
    let one = interpret(r#"ExportString[Framed["a"], "SVG"]"#).unwrap();
    let (_, h) = svg_dims(&one);
    let svg = interpret(
      r#"ExportString[Column[{Framed["a"], Framed["b"]}, Left, 0], "SVG"]"#,
    )
    .unwrap();
    assert_eq!(svg_dims(&svg).1, 2.0 * h, "{svg}");
  }
}

mod typeset_row_items_in_text_face {
  use super::*;

  fn root_height(svg: &str) -> f64 {
    let start = svg.find("height=\"").unwrap() + "height=\"".len();
    let end = start + svg[start..].find('"').unwrap();
    svg[start..end].parse().unwrap()
  }

  /// A `Row` drawn as a picture because one of its items is a matrix keeps
  /// the face of a `Style` around the `Column` it sits in: its label used
  /// to fall back to the monospace face of typeset output.
  #[test]
  fn styled_column_row_with_matrix_takes_the_font_family() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Style[Column[{Row[{q, " = ", MatrixForm[{{1, 2}, {3, 4}}]}]}], FontFamily -> "Times"], "SVG"]"#,
    )
    .unwrap();
    assert!(svg.contains(">q<"), "{svg}");
    assert!(!svg.contains("monospace"), "{svg}");
    assert!(svg.contains("Times"), "{svg}");
  }

  /// `Text[Column[{Grid[…], …}]]` sets a typeset `Row` cell of the grid
  /// (`Row[{label, " = ", MatrixForm[m]}]`) in the proportional text face,
  /// like the plain text cells around it.
  #[test]
  fn text_column_grid_row_with_matrix_uses_text_face() {
    clear_state();
    for code in [
      r#"ExportString[Text[Column[{Grid[{{Row[{q, " = ", MatrixForm[{{1, 2}, {3, 4}}]}]}}], "x"}]], "SVG"]"#,
      r#"ExportString[Style[Column[{Grid[{{Row[{q, " = ", MatrixForm[{{1, 2}, {3, 4}}]}]}}], "x"}], FontFamily -> "sans-serif"], "SVG"]"#,
    ] {
      let svg = interpret(code).unwrap();
      assert!(svg.contains(">q<"), "{code}: {svg}");
      assert!(!svg.contains("monospace"), "{code}: {svg}");
    }
  }

  /// A `Row` cell holding a matrix that is wider than its fixed `ItemSize`
  /// column breaks between its items, so the matrix goes under its label
  /// instead of running into the next column; one that fits stays on one
  /// line.
  #[test]
  fn row_cell_wraps_to_fixed_item_size_width() {
    clear_state();
    let grid = |w: u32| {
      format!(
        r#"ExportString[Text[Grid[{{{{Row[{{q, " = ", MatrixForm[{{{{-1, -2, -2}}, {{2, 1, 2}}, {{2, 2, 3}}}}]}}], "next"}}}}, ItemSize -> {{{{{w}, 5}}}}]], "SVG"]"#
      )
    };
    let wrapped = interpret(&grid(7)).unwrap();
    let single = interpret(&grid(20)).unwrap();
    // The label's line adds about one text line pitch (scaled down with
    // the matrix to the column's width) to the cell's height.
    assert!(
      root_height(&wrapped) > root_height(&single) + 12.0,
      "wrapped: {wrapped}\nsingle: {single}"
    );
    // The label and the matrix are each still drawn once.
    assert_eq!(wrapped.matches(">q<").count(), 1, "{wrapped}");
    assert_eq!(wrapped.matches(">-1<").count(), 1, "{wrapped}");
  }

  /// A `Row` of plain text is one line of text: a column of such rows
  /// stacks at the pitch of the same lines written as single strings,
  /// in the monospace typeset face and in the `Text[…]` face alike,
  /// instead of setting each row apart like a picture.
  #[test]
  fn one_line_text_rows_stack_like_text_lines() {
    clear_state();
    for wrap in ["", "Text@"] {
      let rows = interpret(&format!(
        r#"ExportString[{wrap}Column[{{Row[{{"Width = ", 12}}], Row[{{"Depth = ", x}}], Row[{{Style["Height", Italic], " = ", 2.5}}]}}], "SVG"]"#
      ))
      .unwrap();
      let lines = interpret(&format!(
        r#"ExportString[{wrap}Column[{{"Width = 12", "Depth = x", "Height = 2.5"}}], "SVG"]"#
      ))
      .unwrap();
      assert_eq!(root_height(&rows), root_height(&lines), "{wrap}: {rows}");
    }
  }

  /// A text `Row` under a picture sits as close to it as a line of text
  /// does, with no picture gap in between.
  #[test]
  fn one_line_text_row_under_a_graphic_has_no_picture_gap() {
    clear_state();
    let column = |last: &str| {
      interpret(&format!(
        r#"ExportString[Text@Column[{{Graphics[Disk[], ImageSize -> 50], {last}}}], "SVG"]"#
      ))
      .unwrap()
    };
    assert_eq!(
      root_height(&column(r#"Row[{"Area = ", 3}]"#)),
      root_height(&column(r#""Area = 3""#))
    );
  }

  /// A `Row` holding a picture is still set apart from its neighbors as
  /// a picture.
  #[test]
  fn rows_holding_a_picture_keep_the_picture_gap() {
    clear_state();
    let row = r#"Row[{"m = ", Graphics[Disk[], ImageSize -> 14]}]"#;
    let height = |items: String| {
      root_height(
        &interpret(&format!(r#"ExportString[Column[{{{items}}}], "SVG"]"#))
          .unwrap(),
      )
    };
    assert_eq!(
      height(format!("{row}, {row}")),
      2.0 * height(row.to_string()) + 6.0
    );
  }
}
