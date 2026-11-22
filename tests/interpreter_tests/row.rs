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

  /// `Row[{…picture…}, ImageSize -> …]`: the option sizes the row instead of
  /// being typeset between its items as a separator.
  #[test]
  fn option_of_a_row_holding_a_picture_is_not_a_separator() {
    clear_state();
    let svg = interpret(
      r#"ExportString[Row[{"ab", Graphics[{Disk[]}, ImageSize -> 20]},
        ImageSize -> {120, 30}], "SVG"]"#,
    )
    .unwrap();
    assert!(!svg.contains("ImageSize"), "option typeset as text: {svg}");
    assert!(svg.starts_with("<svg width=\"120\""), "row width: {svg}");
    // A real separator is still set between the items.
    let svg = interpret(
      r#"ExportString[Row[{"ab", Graphics[{Disk[]}, ImageSize -> 20]}, "|"],
        "SVG"]"#,
    )
    .unwrap();
    assert!(svg.contains(">|<"), "separator missing: {svg}");
  }

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

  /// A rule between two pictures is typeset as the pictures around an arrow
  /// inside a row, not echoed as `-Graphics- -> -Graphics-`.
  #[test]
  fn row_of_rules_between_pictures() {
    clear_state();
    let result = interpret_with_stdout(
      r#"g = ArrayPlot[{{1, 2}}]; Row[{g -> g, g -> g}, Spacer[3]]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(!svg.contains("GraphicsBox"), "{svg}");
    assert!(!svg.contains("-Graphics-"), "{svg}");
    assert!(svg.contains("\u{2192}"), "{svg}");
  }

  #[test]
  fn grid_cell_row_of_rules_between_pictures() {
    clear_state();
    let result = interpret_with_stdout(
      r#"g = ArrayPlot[{{1, 2}}]; Grid[{{Text["rule:"], Row[Map[# -> # &, {g, g}], Spacer[3]]}}]"#,
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(!svg.contains("GraphicsBox"), "{svg}");
    assert!(!svg.contains("-Graphics-"), "{svg}");
  }

  /// A row inherits its `Style` the way a column does, and an
  /// `Invisible[…]` item holds its width without being painted.
  #[test]
  fn styled_row_and_invisible_item() {
    clear_state();
    let styled =
      interpret_with_stdout(r#"Style[Row[{"a", "bb"}], 40, Red]"#).unwrap();
    assert_eq!(styled.result, "-Graphics-");
    let svg = styled.graphics.unwrap();
    assert_eq!(svg.matches(">a</").count(), 1, "{svg}");
    assert!(svg.contains("font-size=\"40\""), "{svg}");
    assert!(svg.contains("fill=\"rgb(255,0,0)\""), "{svg}");

    let hidden = interpret_with_stdout(r#"Row[{"a", Invisible["bcd"], "e"}]"#)
      .unwrap()
      .graphics
      .unwrap();
    let shown = interpret_with_stdout(r#"Row[{"a", "bcd", "e"}]"#)
      .unwrap()
      .graphics
      .unwrap();
    assert!(!hidden.contains("Invisible"), "{hidden}");
    assert!(hidden.contains("fill=\"none\""), "{hidden}");
    let dims = |svg: &str| {
      svg
        .split_once('>')
        .map(|(head, _)| head.to_string())
        .unwrap()
    };
    assert_eq!(dims(&hidden), dims(&shown));
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
    // The styled run is a tspan carrying the color inside the text.
    assert!(svg.contains(">hey<"));
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
    assert!(svg.contains(">Hey there<"));
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

  /// A `\n` inside a string item breaks the row onto a new line, also when
  /// the item is wrapped in `Style`.
  #[test]
  fn row_newline_in_string_stacks_lines() {
    clear_state();
    let height = |code: &str| -> f64 {
      let result = interpret_with_stdout(code).unwrap();
      let svg = result.graphics.unwrap();
      svg
        .split("height=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap()
    };
    let one = height("Row[{\"ab\", \" = \", x}]");
    let two = height("Row[{\"ab\\n\", \" = \", x}]");
    let styled = height("Row[{Style[\"ab\\n\", 12], \" = \", x}]");
    assert!(two > one * 1.5, "one = {one}, two = {two}");
    assert!(styled > one * 1.5, "one = {one}, styled = {styled}");
  }

  /// A newline in the middle of a prose string (a blank line between
  /// paragraphs) also breaks the row, while the one-character-per-line
  /// stack of a typeset fraction stays one block.
  #[test]
  fn row_newline_inside_prose_string_stacks_lines() {
    clear_state();
    let height = |code: &str| -> f64 {
      let result = interpret_with_stdout(code).unwrap();
      let svg = result.graphics.unwrap();
      svg
        .split("height=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap()
    };
    let one = height("Row[{\"has \", x, \" red balls\"}]");
    let prose =
      height("Row[{\"has \", x, \" red balls.\\n\\nProbability for\"}]");
    assert!(prose > one * 2.5, "one = {one}, prose = {prose}");
    let fraction = height("Row[{\"y = \", \"1\\n-\\n3\"}]");
    assert!(fraction < one * 3.0, "one = {one}, fraction = {fraction}");
  }

  /// A product whose string factor carries newlines (an implicit
  /// multiplication such as `Row[…] "\n\n  "`) displays its factors
  /// space-separated, so the string breaks the row and the other factor
  /// is drawn on the following line instead of being swallowed.
  #[test]
  fn row_product_with_newline_string_factor_breaks_line() {
    clear_state();
    let svg = |code: &str| -> String {
      interpret_with_stdout(code).unwrap().graphics.unwrap()
    };
    let height = |svg: &str| -> f64 {
      svg
        .split("height=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap()
    };
    let one = svg("Row[{\"ab\", Row[{\"= \", 4}]}]");
    let product = svg("Row[{\"ab\", Row[{\"= \", 4}] \" \\n\\n  \"}]");
    // The string's two newlines put the row factor two lines further down.
    assert!(
      height(&product) > height(&one) * 2.5,
      "one = {}, product = {}",
      height(&one),
      height(&product)
    );
    assert!(product.contains(">4</text>"), "{product}");
    // A product of plain factors still stays on one line.
    let flat = svg("Row[{\"ab\", \"c\" x}]");
    assert!(height(&flat) < height(&one) * 1.5, "{flat}");
  }

  /// A `Subscript` under a `Style` color sits its script as a nested
  /// `<tspan>`, which resvg misplaces (lifting the base off the line); the
  /// row flattens it into sibling spans, each with its inherited style.
  #[test]
  fn row_styled_subscript_flattens_nested_shift_tspan() {
    clear_state();
    let svg = interpret_with_stdout(
      "Row[{x, Style[Style[Subscript[\"log\", 2], Red, 16], Bold], y}]",
    )
    .unwrap()
    .graphics
    .unwrap();
    let text = svg
      .split("<text")
      .find(|t| t.contains("log"))
      .expect("text with log");
    let text = &text[..text.find("</text>").unwrap()];
    // No span opens inside another one.
    let mut depth = 0;
    let mut max_depth = 0;
    for part in text.split('<').skip(1) {
      if part.starts_with("tspan") {
        depth += 1;
        max_depth = max_depth.max(depth);
      } else if part.starts_with("/tspan") {
        depth -= 1;
      }
    }
    assert_eq!(max_depth, 1, "{text}");
    // The shifted script keeps the color and resolves its relative size.
    let script = text
      .split("<tspan")
      .find(|t| t.contains("baseline-shift=\"sub\""))
      .expect("subscript span");
    assert!(script.contains("fill=\"rgb(255,0,0)\""), "{script}");
    assert!(script.contains("font-size=\"11.2\""), "{script}");
  }

  // ToBoxes hands back wolframscript's row template; Woxi's own renderers
  // still stack the lines of a newline string (see the SVG tests above).
  #[test]
  fn traditional_form_row_newline_boxes_are_a_row_template() {
    clear_state();
    assert_eq!(
      interpret("ToBoxes[TraditionalForm[Row[{\"a\\n\", x, \" = \", y}]]]")
        .unwrap(),
      "TagBox[FormBox[TemplateBox[{\"a\\n\", x, \" = \", y}, RowDefault], \
       TraditionalForm], TraditionalForm, Editable -> True]"
    );
  }

  // The row template draws as its parts, a string part as its text.
  #[test]
  fn row_template_box_draws_its_parts() {
    clear_state();
    for code in [
      "ExportString[DisplayForm[TemplateBox[{\"\\\"abc\\\"\", \"x\"}, \
       \"RowDefault\"]], \"SVG\"]",
      "ExportString[Graphics[Text[DisplayForm[TemplateBox[{\"\\\"abc\\\"\", \
       \"x\"}, \"RowDefault\"]]]], \"SVG\"]",
    ] {
      let svg = interpret(code).unwrap();
      assert!(svg.contains("abc"), "{svg}");
      assert!(!svg.contains("&quot;"), "{svg}");
      assert!(!svg.contains("RowDefault"), "{svg}");
    }
  }

  // A `Row` nested inside a TraditionalForm expression (as a left-hand
  // side `P(x)` of an equation) draws as its parts, not as the text of its
  // row template.
  #[test]
  fn traditional_form_nested_row_draws_its_parts() {
    clear_state();
    for code in [
      "ExportString[TraditionalForm[Row[{Style[\"P\", Italic], \"(\", x,        \")\"}] == x^2 + 1], \"SVG\"]",
      "ExportString[TraditionalForm[f[Row[{a, b}]]], \"SVG\"]",
      "ExportString[TraditionalForm[Row[{a, b}] + 1], \"SVG\"]",
    ] {
      let svg = interpret(code).unwrap();
      assert!(!svg.contains("TemplateBox"), "{code}: {svg}");
      assert!(!svg.contains("RowDefault"), "{code}: {svg}");
    }
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

  /// A grid cell whose text holds line breaks (also nested inside `Style`
  /// and `Row`) is drawn as one line per `<tspan>` in a taller row.
  #[test]
  fn grid_cell_newlines_stack_lines() {
    clear_state();
    let svg = |code: &str| -> String {
      interpret_with_stdout(code).unwrap().graphics.unwrap()
    };
    let height = |svg: &str| -> f64 {
      svg
        .split("height=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap()
    };
    let one = svg("Grid[{{\"ab\"}}]");
    let plain = svg("Grid[{{\"a\\nb\"}}]");
    let nested = svg(
      "Grid[{{Text[Style[Row[{Style[\"x\", Red], Style[\"\\ny\\n\", 18]}]]]}}]",
    );
    assert!(height(&plain) > height(&one) * 1.8);
    assert!(height(&nested) > height(&one) * 2.5);
    assert_eq!(plain.matches("<tspan x=").count(), 2);
    assert_eq!(nested.matches("<tspan x=").count(), 3);
    // Every line is well-formed on its own.
    assert_eq!(
      nested.matches("<tspan").count(),
      nested.matches("</tspan>").count()
    );
  }

  /// Each line of a stacked `<text>` repeats the text's central baseline:
  /// resvg (Woxi Studio's renderer) resolves `dominant-baseline` only
  /// against the direct parent, so a colored run nested inside a line
  /// otherwise rose half a line above its row (into the cell above).
  #[test]
  fn stacked_lines_repeat_central_baseline() {
    clear_state();
    let svg = |code: &str| -> String {
      interpret_with_stdout(code).unwrap().graphics.unwrap()
    };
    // The lines are `<tspan>`s of one `<text>`, or (a column in the text
    // face of `Text[…]`) `<text>` elements of their own.
    let line_baselines = |svg: &str, baseline: &str| {
      let mut lines: Vec<&str> = svg
        .split("<tspan x=")
        .skip(1)
        .map(|rest| rest.split('>').next().unwrap())
        .collect();
      if lines.len() < 2 {
        lines = svg
          .split("<text ")
          .skip(1)
          .map(|rest| rest.split('>').next().unwrap())
          .collect();
      }
      assert!(lines.len() >= 2, "{svg}");
      let attr = format!("dominant-baseline=\"{baseline}\"");
      assert!(lines.iter().all(|attrs| attrs.contains(&attr)), "{svg}");
    };
    // Grid cell holding a column of colored rows.
    line_baselines(
      &svg(
        "Grid[{{Text[Column[{Style[Row[{\"a = \", 1}], Red], \
         Style[Row[{\"b = \", 2}], Blue]}]]}}]",
      ),
      "central",
    );
    // Text primitive with a styled run on each line.
    line_baselines(
      &svg(
        "Graphics[Text[Style[Row[{Style[\"a\", Red], \"\\n\", \
         Style[\"b\", Blue]}]], {0, 0}]]",
      ),
      "central",
    );
  }

  fn root_size(svg: &str) -> (f64, f64) {
    let attr = |name: &str| -> f64 {
      svg
        .split(&format!("{name}=\""))
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap()
    };
    (attr("width"), attr("height"))
  }

  /// A plot in a fixed-size `Pane` shrinks to fit the pane, keeping its
  /// whole drawing, instead of being cut off at the pane's edges.
  #[test]
  fn pane_shrinks_plot_to_fit() {
    clear_state();
    let svg = interpret_with_stdout("Pane[Plot[x, {x, 0, 1}], {120, 80}]")
      .unwrap()
      .graphics
      .unwrap();
    let (w, h) = root_size(&svg);
    assert!(w <= 120.0 && h <= 80.0, "{w}x{h}");
    // The plot's own 360 x 221 drawing (as in wolframscript), scaled down
    // whole.
    assert!(svg.contains("viewBox=\"0 0 3600 2210\""));
    assert!(!svg.contains("paneClip"));
  }

  /// A grid made of pictures paints its own `Background`.
  #[test]
  fn picture_grid_paints_background() {
    clear_state();
    let svg = interpret_with_stdout(
      "Grid[{{Graphics[Circle[]], Graphics[Disk[]]}}, Background -> Yellow]",
    )
    .unwrap()
    .graphics
    .unwrap();
    assert!(svg.contains("fill=\"rgb(255,255,0)\""));
  }

  /// In a fixed-width `Pane`, a nested legend grid holding automatically
  /// sized pictures is squeezed so the columns beside it stay inside the
  /// pane rather than being pushed out of view.
  #[test]
  fn pane_squeezes_nested_legend_grid() {
    clear_state();
    let svg = interpret_with_stdout(
      "Pane[Grid[{{Pane[Plot[x, {x, 0, 1}], {200, 120}], \
       Pane[Plot[x, {x, 0, 1}], {200, 120}]}, \
       {Grid[{{Graphics[Line[{{0, 0}, {1, 0}}], AspectRatio -> 1/8], \
       \"label\"}}]}}], {420, 200}]",
    )
    .unwrap()
    .graphics
    .unwrap();
    let (w, _) = root_size(&svg);
    assert!(w <= 420.0, "{w}");
    // Every top-level picture starts inside the pane.
    for part in svg.split("<svg x=\"").skip(1) {
      let x: f64 = part.split('"').next().unwrap().parse().unwrap();
      assert!(x < 420.0, "picture starts at {x}");
    }
    // The legend line was redrawn narrower, not left at its 360pt default.
    assert!(!svg.contains("width=\"360\" height=\"45\""));
  }
}

/// Regression: a `Row` of `Button`s built with `Map` and placed in a `Grid`
/// cell was written out as the buttons' source text instead of a row of
/// button plates.
#[test]
fn row_of_buttons_in_grid_cell_draws_plates() {
  let svg = interpret(
    r#"ExportString[Grid[{{Row[Map[Button[Grid[{{#}}], 1] &, {"a", "b"}]]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(!svg.contains("Button["), "button source leaked: {svg}");
  assert_eq!(
    svg.matches("<rect").count() - svg.matches("id=\"woxi-btn-").count(),
    2,
    "one plate per button: {svg}"
  );
}

/// A text-labeled `Button[label, action]` drawn as a `Row` item (an
/// on-screen keyboard of letter keys) carries the invisible hit region a
/// live widget presses, encoding the button's own action — not just an
/// inert plate.
#[test]
fn row_of_text_buttons_carries_pressable_hit_regions() {
  let svg = interpret(
    r#"ExportString[Row[Map[Button[#, typed = typed <> #] &, {"p", "q", "r"}]], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert_eq!(
    svg.matches("id=\"woxi-btn-").count(),
    3,
    "one hit region per key: {svg}"
  );
  for key in ["p", "q", "r"] {
    let action = format!("typed = StringJoin[typed, \"{key}\"]");
    let hex: String = action
      .bytes()
      .flat_map(|b| {
        let digits = b"0123456789abcdef";
        [
          digits[(b >> 4) as usize] as char,
          digits[(b & 15) as usize] as char,
        ]
      })
      .collect();
    assert!(
      svg.contains(&format!("id=\"woxi-btn-{hex}\"")),
      "key {key} should press {action:?}: {svg}"
    );
  }
}

/// A `Button` without an action has nothing to press, so it gets no hit
/// region.
#[test]
fn row_of_action_less_buttons_has_no_hit_regions() {
  let svg =
    interpret(r#"ExportString[Row[{Button["a"], Button["b"]}], "SVG"]"#)
      .unwrap();
  assert!(!svg.contains("woxi-btn-"), "{svg}");
}

/// Regression: SVG trims edge whitespace unless told not to, so a styled
/// `Row` part with a trailing or doubled space (`Style["sum   = ", 14]`)
/// lost those spaces — its `<tspan>` wrapper hid them from the check that
/// adds `xml:space="preserve"`.
#[test]
fn styled_row_part_keeps_edge_and_doubled_spaces() {
  let svg = interpret(
    r#"ExportString[Row[{Style["sum   = ", 14], Style["7", 16, Bold]}], "SVG"]"#,
  )
  .unwrap();
  let part = svg
    .lines()
    .find(|l| l.contains("sum   = "))
    .unwrap_or_else(|| panic!("the part's spaces must be kept: {svg}"));
  assert!(part.contains("xml:space=\"preserve\""), "{part}");
  // A part without edge or doubled spaces needs no preserve attribute.
  let seven = svg.lines().find(|l| l.contains(">7<")).expect("7 part");
  assert!(!seven.contains("xml:space"), "{seven}");
}

/// Regression: a `Panel` in a `Grid` cell (a readout beside plots) printed
/// its `Panel[…]` source text instead of the light-gray box around its
/// content.
#[test]
fn panel_in_grid_cell_draws_box() {
  let svg = interpret(
    r#"ExportString[Grid[{{Panel[Row[{"x", " = ", "1"}]], Panel["y"]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(!svg.contains("Panel"), "panel source leaked: {svg}");
  assert_eq!(
    svg.matches("fill=\"rgb(245,245,245)\"").count(),
    2,
    "one box per panel: {svg}"
  );
}

/// A `Pane[…]` item of a `Row` inside a grid cell only sizes its content
/// (a fixed-width numeric readout); the cell shows the content, not the
/// `Pane[…]` call.
#[test]
fn pane_item_of_row_in_grid_cell_shows_content() {
  let svg = interpret(
    r#"ExportString[Grid[{{"Area:", Row[{Pane[NumberForm[0.744631, {6, 5}], {50, 32}, ImageSizeAction -> "ShrinkToFit", Alignment -> {Right, Center}], " r"}]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(!svg.contains("Pane"), "pane source leaked: {svg}");
  assert!(svg.contains("0.74463"), "readout missing: {svg}");
}

mod dynamic_release_in_frames {
  use super::*;

  // A Dynamic inside a Panel is shown (and evaluated) in document order,
  // after its Initialization; a later Dynamic reads what it set.
  #[test]
  fn panel_dynamic_runs_before_later_dynamics() {
    clear_state();
    let result = interpret_with_stdout(
      "Clear[a, b, sel]; Column[{Panel[Dynamic[a = {sel}, Initialization :> (sel = 7)]], \
       Dynamic[First[a]]}]",
    )
    .unwrap();
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result.graphics.unwrap().contains(">7</text>"));
  }
}

mod labeled_text_layout {
  use super::*;

  /// A caption set on a sized pane of laid-out text is a drawn picture in
  /// the notebook, not the call echoed back as source text.
  #[test]
  fn labeled_pane_of_text_column_draws() {
    clear_state();
    let r = interpret_with_stdout(
      r#"Style[Labeled[Pane[Text[Column[{Row[{"a", "b"}], "c"}]], {120, 60}], Text["cap"], Top], 24]"#,
    )
    .unwrap();
    assert!(
      r.result.starts_with("-Graphics-") || r.result.contains("<svg"),
      "Labeled text layout was not drawn: {}",
      r.result
    );
    assert!(!r.result.contains("Labeled["), "{}", r.result);
  }
}

mod traditional_row_layout {
  use super::*;

  fn svg(code: &str) -> String {
    clear_state();
    interpret(&format!("ExportString[{code}, \"SVG\"]")).unwrap()
  }

  /// The plain contents of the `<text>` elements, in document order.
  fn texts(svg: &str) -> Vec<String> {
    svg
      .split("<text")
      .skip(1)
      .filter_map(|t| {
        let body = t.split_once('>')?.1;
        Some(body.split('<').next()?.to_string())
      })
      .collect()
  }

  /// `RawBoxes["("]` is an already typeset box: inside a TraditionalForm
  /// row it draws as a bare parenthesis, not as `RawBoxes[(]`.
  #[test]
  fn raw_boxes_item_draws_its_boxes() {
    let out = svg(
      "Text[TraditionalForm[Row[{\"at \", RawBoxes[\"(\"], 1, \", \", 2, \
       RawBoxes[\")\"]}]]]",
    );
    assert!(!out.contains("RawBoxes"), "{out}");
    let t = texts(&out);
    assert!(t.iter().any(|s| s == "("), "{t:?}");
    assert!(t.iter().any(|s| s == ")"), "{t:?}");
  }

  /// A product with a line-breaking string factor flows into a
  /// TraditionalForm row factor by factor: its newline breaks the row and
  /// what follows continues on the next line.
  #[test]
  fn product_with_newline_string_breaks_traditional_row() {
    let out = svg(
      "Text[TraditionalForm[Row[{\"The \", \"real\" \" roots are \\n\", x, \
       \" = 1\"}]]]",
    );
    assert!(out.starts_with("<svg width=\"") && out.contains("height=\"44\""));
    let t = texts(&out);
    let pos = |s: &str| t.iter().position(|x| x == s).unwrap();
    assert!(pos("real") < pos(" roots are "), "{t:?}");
    assert!(pos(" roots are ") < pos(" = 1"), "{t:?}");
  }

  /// A polynomial item of a row is typeset in descending powers, the same
  /// as on its own: `x² + 2 x`, not `2 x + x²`.
  #[test]
  fn polynomial_item_keeps_descending_powers() {
    let t = texts(&svg("Row[{TraditionalForm[x^2 + 2 x], \"a\"}]"));
    let visible: Vec<&str> = t
      .iter()
      .map(String::as_str)
      .filter(|s| !s.trim().is_empty())
      .collect();
    assert_eq!(visible, ["x", "2", "+", "2", "x", "a"]);
  }
}

/// Row items are typeset like StandardForm values: an inert call of an
/// undefined head stacks the fractions among its arguments, and the
/// constants `I`, `Pi`, … are set as their glyphs inside stacked values.
mod row_standard_typesetting {
  use super::*;

  fn svg(code: &str) -> String {
    clear_state();
    interpret(&format!("ExportString[{code}, \"SVG\"]")).unwrap()
  }

  fn texts(svg: &str) -> Vec<String> {
    svg
      .split("<text")
      .skip(1)
      .filter_map(|t| {
        let body = t.split_once('>')?.1;
        let body = body
          .strip_prefix("<tspan")
          .map_or(body, |b| b.split_once('>').map_or(b, |(_, rest)| rest));
        Some(body.split('<').next()?.to_string())
      })
      .filter(|s| !s.trim().is_empty())
      .collect()
  }

  #[test]
  fn undefined_head_call_stacks_its_fraction_argument() {
    let t = texts(&svg("Row[{g[3/8], \"=\", x}]"));
    assert!(t.iter().any(|s| s == "3"), "{t:?}");
    assert!(t.iter().any(|s| s == "8"), "{t:?}");
    assert!(!t.iter().any(|s| s.contains("3/8")), "{t:?}");
  }

  #[test]
  fn styled_undefined_head_call_stacks_its_fraction_argument() {
    let t = texts(&svg("Style[Row[{g[13/42], \"=\", 5/7}], 24]"));
    assert!(t.iter().any(|s| s == "13"), "{t:?}");
    assert!(t.iter().any(|s| s == "42"), "{t:?}");
  }

  #[test]
  fn imaginary_unit_in_a_stacked_value_is_the_glyph() {
    for code in [
      "Row[{I/2}]",
      "Row[{5/7 + (4 I)/(7 Sqrt[3])}]",
      "Style[Row[{5/7 + (4 I)/(7 Sqrt[3])}], 24]",
    ] {
      let t = texts(&svg(code));
      assert!(!t.iter().any(|s| s == "I"), "{code}: {t:?}");
    }
  }

  #[test]
  fn pi_in_a_stacked_row_item_is_the_glyph() {
    let t = texts(&svg("Row[{Pi/2, x}]"));
    assert!(t.iter().any(|s| s == "\u{03C0}"), "{t:?}");
    assert!(!t.iter().any(|s| s == "Pi"), "{t:?}");
  }

  #[test]
  fn string_literal_is_not_replaced_by_a_glyph() {
    let t = texts(&svg("Row[{g[\"Pi\", 1/2]}]"));
    assert!(t.iter().any(|s| s.contains("Pi")), "{t:?}");
  }
}

/// Regression: a `Row` holding a nested `Grid` (a binomial-style `N` over
/// `k` between parentheses) inside a framed `Grid` cell printed the inner
/// grid's `Grid[{{…}}]` source text instead of stacking its rows.
#[test]
fn row_with_nested_grid_in_grid_cell_stacks_rows() {
  let svg = interpret(
    r#"ExportString[Grid[{{x, Row[{a, "(", Grid[{{"n"}, {"k"}}], ")"}]}}, Frame -> True], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(!svg.contains("Grid["), "grid source leaked: {svg}");
  let y_of = |s: &str| -> f64 {
    let at = svg.find(&format!(">{s}</text>")).unwrap();
    let head = &svg[..at];
    let tag = &head[head.rfind("<text").unwrap()..];
    let y = tag.split(" y=\"").nth(1).unwrap();
    y.split('"').next().unwrap().parse().unwrap()
  };
  assert!(y_of("n") < y_of("k"), "n is not above k: {svg}");
}

/// The same row typeset in TraditionalForm stacks the grid's rows too,
/// instead of writing `Grid[` around a parenthesized matrix.
#[test]
fn traditional_row_with_nested_grid_stacks_rows() {
  clear_state();
  let svg = interpret(
    r#"ExportString[Grid[{{x, TraditionalForm[Row[{a, Grid[{{"n"}, {"k"}}]}]]}}, Frame -> True], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(!svg.contains(">Grid<"), "grid head leaked: {svg}");
  assert!(!svg.contains(">(<"), "matrix parentheses leaked: {svg}");
}

/// A `Button[label, action, Appearance -> None]` in a `Grid` cell draws only
/// its styled label — no plate — while still carrying the hit region a live
/// widget presses; a default-appearance neighbor keeps its plate.
#[test]
fn appearance_none_button_in_grid_draws_label_without_plate() {
  clear_state();
  let svg = interpret(
    r#"ExportString[Grid[{{Button[Style["<", Large, Bold], k = k + 1, Appearance -> None], Button["ok", k = 0]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert_eq!(
    svg.matches("id=\"woxi-btn-").count(),
    2,
    "both buttons are pressable: {svg}"
  );
  assert_eq!(
    svg.matches("rx=\"4\"").count(),
    1,
    "only the default button draws a plate: {svg}"
  );
  let label = svg.find("&lt;</text>").expect("arrow label drawn");
  let tag = &svg[svg[..label].rfind("<text").unwrap()..label];
  assert!(tag.contains("font-weight=\"bold\""), "bold label: {tag}");
  assert!(tag.contains("sans-serif"), "button face font: {tag}");
}

/// Regression: a styled `Row` (`Style[Row[{…, Spacer[n], …}], size]`)
/// pushes its style onto every item, and the resulting
/// `Style[Spacer[n], size]` was typeset as text — dozens of points wide —
/// instead of the `n` points of blank the spacer stands for.
#[test]
fn styled_spacer_keeps_its_width() {
  clear_state();
  let width = |code: &str| -> f64 {
    let svg = interpret(&format!("ExportString[{code}, \"SVG\"]")).unwrap();
    let w = svg.split(" width=\"").nth(1).unwrap();
    w.split('"').next().unwrap().parse().unwrap()
  };
  let plain = width("Row[{x, Spacer[10], b}]");
  assert_eq!(width("Row[{x, Style[Spacer[10], 11], b}]"), plain);
  assert_eq!(width("Row[{x, Style[Spacer[10], Red], b}]"), plain);
  // The styled row's items are set at 11 pt, narrower than the default
  // 14 pt, so the whole row is no wider than the unstyled one.
  for opts in ["", ", Frame -> True"] {
    let styled = width(&format!("Style[Row[{{x, Spacer[10], b}}{opts}], 11]"));
    assert!(styled <= plain, "{opts}: {styled} > {plain}");
  }
}

/// A relation separator is set as an operator, with a space on each side,
/// while a punctuation separator stays tight.
#[test]
fn row_relation_separator_gets_operator_spacing() {
  clear_state();
  let svg = interpret(r#"ExportString[Row[{x, 2}, "="], "SVG"]"#).unwrap();
  assert!(svg.contains("> = </text>"), "unspaced `=`: {svg}");
  let svg =
    interpret(r#"ExportString[Row[{x, 2}, "\[Rule]"], "SVG"]"#).unwrap();
  assert!(svg.contains("> \u{2192} </text>"), "unspaced arrow: {svg}");
  let svg = interpret(r#"ExportString[Row[{x, 2}, ","], "SVG"]"#).unwrap();
  assert!(svg.contains(">,</text>"), "spaced comma: {svg}");
}

/// A `Style` around a row with a separator styles the separator too.
#[test]
fn styled_row_styles_its_separator() {
  clear_state();
  let svg =
    interpret(r#"ExportString[Style[Row[{x, 2.5}, "="], 14, Orange], "SVG"]"#)
      .unwrap();
  assert!(
    svg.contains(r#"fill="rgb(255,128,0)"> = </tspan>"#),
    "separator not orange: {svg}"
  );
  // The separator's own style wins over the enclosing one.
  let svg = interpret(
    r#"ExportString[Style[Row[{x, y}, Style[" | ", Gray]], Orange], "SVG"]"#,
  )
  .unwrap();
  assert!(
    svg.contains(r#"fill="rgb(128,128,128)"> | </tspan>"#),
    "separator not gray: {svg}"
  );
}

/// Regression: a `Button` in a `Grid` cell drew a `Style`d label
/// (`Style[n, "Title"]`, `Style[k, 30, Bold, Red]`) as plain 14-point
/// black text, dropping the label's font size, color and weight.
#[test]
fn styled_button_label_in_grid_keeps_its_style() {
  let svg = interpret(
    r#"ExportString[Grid[{{Button[Style[7, "Title"], x = 1]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.contains("font-size=\"44\""), "Title size lost: {svg}");

  let svg = interpret(
    r#"ExportString[Grid[{{Button[Style["go", 30, Bold, Red], x = 1, ImageSize -> {120, 60}]}}], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.contains("font-size=\"30\""), "size lost: {svg}");
  assert!(svg.contains("font-weight=\"bold\""), "weight lost: {svg}");
  assert!(
    !svg.contains("fill=\"#000000\" text-anchor"),
    "color lost: {svg}"
  );

  // A plain label still draws as the default 14-point text.
  let svg =
    interpret(r#"ExportString[Grid[{{Button["go", x = 1]}}], "SVG"]"#).unwrap();
  assert!(svg.contains("font-size=\"14\""), "{svg}");
}

/// A relation separator of a `Row` inside a `Graphics` `Text` label is
/// spaced like the operator, as in a standalone `Row`.
#[test]
fn row_relation_separator_spaced_in_graphics_text() {
  clear_state();
  let svg = interpret(
    r#"ExportString[Graphics[Text[Row[{v, 2.5}, "="], {0, 0}]], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.contains("</tspan> = 2.5</text>"), "unspaced `=`: {svg}");
  let svg = interpret(
    r#"ExportString[Graphics[Text[Row[{v, 2.5}, ","], {0, 0}]], "SVG"]"#,
  )
  .unwrap();
  assert!(svg.contains("</tspan>,2.5</text>"), "spaced comma: {svg}");
}

/// `Text[expr]` sets `expr` in plain text: an enclosing `Bold` does not
/// reach the `Text` items of a styled row (their size still does), while a
/// `Style` inside the `Text` applies.
#[test]
fn text_items_of_bold_row_are_plain() {
  clear_state();
  let svg = interpret(
    r#"ExportString[Style[Row[{Text["name"], " ", Text[Subscript["CO", 2]]}], 18, Bold], "SVG"]"#,
  )
  .unwrap();
  let text_of = |label: &str| {
    svg
      .lines()
      .find(|l| l.contains("<text") && l.contains(label))
      .unwrap_or_else(|| panic!("no {label} text: {svg}"))
      .to_string()
  };
  for label in [">name</text>", ">CO<tspan"] {
    let line = text_of(label);
    assert!(line.contains("font-size=\"18\""), "size lost: {line}");
    assert!(!line.contains("font-weight=\"bold\""), "bold: {line}");
  }

  let svg = interpret(
    r#"ExportString[Style[Row[{Text[Style["name", Bold]], "x"}], 18], "SVG"]"#,
  )
  .unwrap();
  // The weight sits on the `<text>` or on a `<tspan>` around the word.
  let line = svg
    .lines()
    .find(|l| l.contains(">name</text>") || l.contains(">name</tspan>"))
    .unwrap_or_else(|| panic!("no name text: {svg}"));
  assert!(
    line.contains("font-weight=\"bold\""),
    "inner Bold lost: {line}"
  );
  assert!(line.contains("font-size=\"18\""), "size lost: {line}");
}

/// The x coordinate of the first `<text>` element with `text-anchor="start"`.
fn first_start_text_x(svg: &str) -> f64 {
  svg
    .split("text-anchor=\"start\"")
    .next()
    .and_then(|head| head.rsplit("<text x=\"").next())
    .and_then(|t| t.split('"').next())
    .and_then(|n| n.parse::<f64>().ok())
    .unwrap_or_else(|| panic!("no left-aligned text: {svg}"))
}

#[test]
fn graphics_row_image_size_label_is_flush_left_in_its_box() {
  // Regression: an `Inset`/`Text` label `Row[{…}, ImageSize -> w]` had its
  // lines centered on the anchor; Wolfram sets them flush left in a box
  // `w` points wide that is centered on the anchor.
  clear_state();
  let svg = interpret(
    "ExportString[Graphics[{Inset[Style[Row[{\"ab cdef\\ngh\"}, \
     ImageSize -> 200], 11], {0, 0}]}, PlotRange -> {{-5, 5}, {-5, 5}}, \
     ImageSize -> 300], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.contains(">ab cdef</tspan>"), "first line: {svg}");
  assert!(svg.contains(">gh</tspan>"), "second line: {svg}");
  // The picture's center is at 149.5; the box's left edge 100 points left.
  let x = first_start_text_x(&svg);
  assert!((x - 49.5).abs() < 1.0, "left edge at {x}: {svg}");

  // A single line hangs from the box's left edge too.
  clear_state();
  let svg = interpret(
    "ExportString[Graphics[{Text[Row[{\"one\"}, ImageSize -> 100], {0, 0}]}, \
     PlotRange -> {{-5, 5}, {-5, 5}}, ImageSize -> 300], \"SVG\"]",
  )
  .unwrap();
  let x = first_start_text_x(&svg);
  assert!((x - 99.5).abs() < 1.0, "left edge at {x}: {svg}");
  assert!(!svg.contains("ImageSize"), "option leaked: {svg}");
}

#[test]
fn graphics_row_image_size_label_wraps_at_its_width() {
  // Text longer than the row's width breaks at word boundaries.
  clear_state();
  let svg = interpret(
    "ExportString[Graphics[{Inset[Style[Row[{\"aaaa aaaa aaaa aaaa aaaa\"}, \
     ImageSize -> 120], 20], {0, 0}]}, PlotRange -> {{-5, 5}, {-5, 5}}, \
     ImageSize -> 400], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.contains(">aaaa aaaa</tspan>"), "not wrapped: {svg}");
  assert!(!svg.contains("aaaa aaaa aaaa"), "wrapped too wide: {svg}");
  let x = first_start_text_x(&svg);
  assert!((x - 139.5).abs() < 1.0, "left edge at {x}: {svg}");
}
